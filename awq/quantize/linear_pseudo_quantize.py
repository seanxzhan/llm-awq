from transformers import AutoProcessor, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from PIL import Image
from collections import defaultdict

import tqdm
import torch
import gc
from torch import nn

from functools import partial

from ..utils.calib_data import get_calib_dataset
from ..utils.calib_data import get_calib_dataset_openvla

# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Calculate the maximum (\alpha) and minimum values (\beta) in the tensor.
    max_val = w.amax(dim=1, keepdim=True)
    assert max_val.dim() == 2 and max_val.size(0) == w.size(0) and max_val.size(1) == 1
    min_val = w.amin(dim=1, keepdim=True)
    assert min_val.dim() == 2 and min_val.size(0) == w.size(0) and min_val.size(1) == 1

    # Calculate the scale factor and zero point.  (Formula 1 & 2)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    assert scales.shape == max_val.shape
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    assert scales.shape == min_val.shape

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    # Dequantize W (pseudo quantization, the inverse transformation of Formula 3)
    w = (w - zeros) * scales
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w

def pseudo_quantize_model_salient_weight_fp16(
    model, enc, w_bit, q_group_size, n_samples=512, seqlen=512, calib_data="pileval"
):
    from .pre_quant import get_blocks, get_named_linears, move_embed
    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model)
    # input_feat = dict()
    # def stat_input_max_hook(m, x, y, name):
    #     if isinstance(x, tuple):
    #         x = x[0]
    #     x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
    #     if name not in input_feat:
    #         input_feat[name] = [x_max]
    #     else:
    #         input_feat[name] += [x_max]

    # hooks = []
    # for i in tqdm.tqdm(range(len(layers)), desc="building input_feat..."):
    #     named_linears = get_named_linears(layers[i])
    #     for n, m in named_linears.items():
    #         hooks.append(
    #             m.register_forward_hook(
    #                 partial(stat_input_max_hook, name=n)))
    if calib_data == "openvla":
        samples = get_calib_dataset_openvla(
            data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
        )
    else:
        samples = get_calib_dataset(
            data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
        )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        if calib_data == "openvla":
            print("running the model with hooks...")
            pbar = tqdm.tqdm(samples)
            for sample in pbar:
                sample = sample.to(next(model.parameters()).device).unsqueeze(0)
                model(inputs_embeds=sample)
                pbar.update()
        else:
            model(samples.to(next(model.parameters()).device))
        # model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    for i in tqdm.tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()

        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        input_feat = {k: [row for row in v[0]] for k, v in input_feat.items()}
        # print(input_feat)
        for n, m in named_linears.items():
            importance = sum(input_feat[n]).float()
            # print(importance)

            _, outlier_indices = torch.topk(importance, int(0.01 * len(importance)))
            # if outlier_indices.dim() != 1:
            #     outlier_indices = outlier_indices.flatten()
            # print(outlier_indices.flatten())
            assert outlier_indices.dim() == 1

            outlier = m.weight.data[:, outlier_indices].clone()
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )
            m.weight.data[:, outlier_indices] = outlier
            m.cpu()

        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

