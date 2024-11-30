from transformers import AutoProcessor, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from PIL import Image

import tqdm
import torch
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
    input_feat = dict()
    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        if name not in input_feat:
            input_feat[name] = [x_max]
        else:
            input_feat[name] += [x_max]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_input_max_hook, name=name)))
    if calib_data == "openvla":
        samples = get_calib_dataset_openvla(
            data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
        )
    else:
        samples = get_calib_dataset(
            data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
        )
    pbar = tqdm.tqdm(samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()


    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            importance = sum(input_feat[n]).float()

            _, outlier_indices = torch.topk(importance, int(0.01 * len(importance)))
            assert outlier_indices.dim() == 1

            outlier = m.weight.data[:, outlier_indices].clone()
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )
            m.weight.data[:, outlier_indices] = outlier
            m.cpu()


