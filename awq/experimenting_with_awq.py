from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from PIL import Image

import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from functools import partial

from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

import gc

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def get_model_size(model: nn.Module, data_width=16, group_size=-1):

    if group_size != -1:
        data_width += (16 + 4) / group_size

    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width


def get_calib_dataset(tokenizer=None, processor=None, n_samples=256, block_size=512):
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in "openvla/openvla-7b" else VicunaV15ChatPromptBuilder,
    )

    vla_dataset = RLDSDataset(
        "/datasets",
        "bridge_orig",
        batch_transform,
        resize_resolution=(224, 224),  # 224 is hard coded, originally tuple(vla.module.config.image_sizes)
        shuffle_buffer_size=100_000,
        image_aug=False,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=1,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation") # TODO: also need to update this dataset load
    # dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to("cuda:0")
        # line = data["text"]
        # line = line.strip()
        # line_encoded = tokenizer.encode(line)
        line_encoded = input_ids
        # print(line_encoded)
        if len(line_encoded) > block_size:
            continue
        sample = line_encoded
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    # print(samples)
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)]


def get_calib_dataset_both(tokenizer=None, processor=None, n_samples=256, block_size=512):
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in "openvla/openvla-7b" else VicunaV15ChatPromptBuilder,
    )

    vla_dataset = RLDSDataset(
        "/datasets",
        "bridge_orig",
        batch_transform,
        resize_resolution=(224, 224),  # 224 is hard coded, originally tuple(vla.module.config.image_sizes)
        shuffle_buffer_size=100_000,
        image_aug=False,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=1,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation") # TODO: also need to update this dataset load
    # dataset = dataset.shuffle(seed=42)

    samples = []
    n_run = 0
    for batch_idx in range(len(dataloader)):
        embed = torch.load(f"/sota/openvla/ckpt/openvla-7b+calib_feat+b1--original/mutlimodal_embeddings/{batch_idx}.pt")
        # embed = embed.to("cuda:0")
        # line = data["text"]
        # line = line.strip()
        # line_encoded = tokenizer.encode(line)
        # print(line_encoded)
        # if len(embed) > block_size:
        #     continue
        sample = embed
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    # print(samples)
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)]


def get_calib_feat(model, tokenizer, processor):
    input_dict = dict()

    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]

    # hooks = []
    # for name, m in model.named_modules():
    #     if isinstance(m, nn.Linear):
    #         hooks.append(m.register_forward_hook(partial(stat_input_max_hook, name=name)))

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_input_max_hook, name=name)))
        # elif isinstance(m, nn.Linear):
        #     hooks.append(
        #         m.register_forward_hook(
        #             partial(stat_input_max_hook, name=name)))
        # elif isinstance(m, nn.Linear):
        #     # Collect input to Linear layers if needed
        #     pass  # If required, add hooks here
        # elif name.endswith('self_attn'):
        #     def attn_output_hook(m, x, y, name):
        #         # y is the output of the self-attention module
        #         if isinstance(y, tuple):
        #             y = y[0]
        #         if name + ".attn_output" not in input_dict:
        #             input_dict[name + ".attn_output"] = [y.cpu()]
        #         else:
        #             input_dict[name + ".attn_output"] += [y.cpu()]
        #     hooks.append(
        #         m.register_forward_hook(
        #             partial(attn_output_hook, name=name)))
        # elif name.endswith('act_fn'):
        #     def mlp_act_hook(m, x, y, name):
        #         if isinstance(y, tuple):
        #             y = y[0]
        #         if name not in input_dict:
        #             input_dict[name] = [y.cpu()]
        #         else:
        #             input_dict[name] += [y.cpu()]
        #     hooks.append(
        #         m.register_forward_hook(
        #             partial(mlp_act_hook, name=name)))


    print("Collecting activation scales...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = get_calib_dataset_both(tokenizer, processor)
    pbar = tqdm.tqdm(samples)
    for sample in pbar:
        sample = sample.to(device)
        # model(sample)
        model(inputs_embeds=sample)

    for hook in hooks:
        hook.remove()

    # print(input_dict.keys())
    return input_dict


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
    max_int = 2**n_bit - 1
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


def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_group_size,
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)


def pseudo_quantize_model_salient_weight_fp16(model, w_bit, q_group_size, input_feat):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############

            # Step 1: Find 1% of the salient weight channels according to importance (hint: use torch.topk())
            _, outlier_indices = torch.topk(importance, int(0.01 * len(importance)))
            assert outlier_indices.dim() == 1

            ############### YOUR CODE ENDS HERE #################

            # Back up the values of the salient weight channels
            outlier = m.weight.data[:, outlier_indices].clone()

            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

            ############### YOUR CODE STARTS HERE ###############

            # Step 2: Restore the 1% salient weight channels to their original FP16 values
            m.weight.data[:, outlier_indices] = outlier

            ############### YOUR CODE ENDS HERE #################


def pseudo_quantize_model_weight_scaleup(model, w_bit, q_group_size, input_feat, scale_factor):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############

            # Step 1: Find 1% of the salient weight channels
            _, outlier_mask = torch.topk(importance, int(0.01 * len(importance)))
            assert outlier_mask.dim() == 1

            ############### YOUR CODE ENDS HERE #################

            # To simulate applying the scale factor, we can simply multiply it before quantization, and then divide by the scale factor after quantization.
            # Scale up the values of the salient weight channels
            m.weight.data[:, outlier_mask] *= scale_factor

            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

            ############### YOUR CODE STARTS HERE ###############

            # Step 2: Scale back down the values of the salient weight channels
            m.weight.data[:, outlier_mask] /= scale_factor

            ############### YOUR CODE ENDS HERE #################


def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


def scale_fcs(fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(fcs.weight.device)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


def auto_scale_block(module, name, w_bit, q_group_size, input_feat):

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):

        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        s_x = x.view(-1, x.shape[-1]).abs().mean(0)

        ############### YOUR CODE STARTS HERE ###############

        # Step 1: Initialize the best_error, best_ratio and best_scales
        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        ############### YOUR CODE ENDS HERE #################

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            # ratio is the \alpha in the formula
            ratio = ratio * 1 / n_grid

            ############### YOUR CODE STARTS HERE ###############

            # Step 2: Calculate the scales by the formula: scales = s_x^ratio
            scales = s_x.pow(ratio).clamp(min=1e-4)
            assert scales.shape == s_x.shape

            ############### YOUR CODE ENDS HERE #################

            scales = scales / (scales.max() * scales.min()).sqrt().view(1, -1)

            for fc in linears2scale:

                scales = scales.to(fc.weight.device)

                # Scale up the values of the weight channels
                fc.weight.mul_(scales)

                fc.weight.data = pseudo_quantize_tensor(fc.weight.data, w_bit, q_group_size)

                ############### YOUR CODE STARTS HERE ###############

                # Step 3: Scale back down the values of the weight channels
                fc.weight.div_(scales)

                ############### YOUR CODE ENDS HERE #################

            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            print(history)
            raise Exception

        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    # attention input
    inp = input_feat[name + ".self_attn.o_proj"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0).unsqueeze(0)
    qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
    final_scales = _search_module_scale(module.self_attn, qkv, inp)
    scale_ln_fcs(module.post_attention_layernorm, qkv, final_scales)

    # attn out
    inp = input_feat[name + ".self_attn.o_proj"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.self_attn.o_proj, [module.self_attn.o_proj], inp)
    scale_fc_fc(module.self_attn.v_proj, module.self_attn.o_proj, final_scales)

    inp = input_feat[name + ".mlp.act_fn"]
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    mlps = [module.mlp.gate_proj, module.mlp.up_proj, module.mlp.down_proj]
    final_scales = _search_module_scale(module.mlp, mlps, inp)
    scale_fcs(mlps, final_scales)


def pseudo_quantize_model_weight_auto_scale(model, w_bit, q_group_size, input_feat):
    from transformers.models.opt.modeling_opt import OPTDecoderLayer

    for name, module in model.named_modules():
        # print(name)
        if isinstance(module, OPTDecoderLayer):
            auto_scale_block(module, name, w_bit, q_group_size, input_feat)

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)


def load_model(model_path="openvla/openvla-7b"):
    return AutoModelForVision2Seq.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")


def load_model_llm_backbone(model):
    return model.language_model


def get_pseudo_quantized_model(model_path="openvla/openvla-7b"):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    vla = load_model()
    model = load_model_llm_backbone(vla)
    model_size = get_model_size(model, data_width=16, group_size=128)  # TODO: is the group_size param accurate?
    print(f"LLM backbone model size: {model_size/MiB:.2f} MiB")

    pseudo_quantize_model_weight(model, w_bit=3, q_group_size=128)
    model_size = get_model_size(model, data_width=3, group_size=128)
    print(f"LLM backbone model size after really simple pseudo quantization: {model_size/MiB:.2f} MiB")

    vla.language_model = model
    return vla


def get_pseudo_salient_quantized_model(model_path="openvla/openvla-7b"):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    vla = load_model()
    model = load_model_llm_backbone(vla)
    model_size = get_model_size(model, data_width=16, group_size=128)  # TODO: is the group_size param accurate?
    print(f"LLM backbone model size: {model_size/MiB:.2f} MiB")

    # pseudo_quantize_model_weight(model, w_bit=3, q_group_size=128)
    input_feat = get_calib_feat(model, tokenizer, processor)  # Note this is input_feat for LLM backbone and not OpenVLA
    pseudo_quantize_model_salient_weight_fp16(model, w_bit=3, q_group_size=128, input_feat=input_feat)
    model_size = get_model_size(model, data_width=3, group_size=128)
    print(f"LLM backbone model size after salient pseudo quantization: {model_size/MiB:.2f} MiB")

    vla.language_model = model
    return vla


def get_awq_scale_quantized_model(model_path="openvla/openvla-7b"):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    vla = load_model()
    model = load_model_llm_backbone(vla)
    model_size = get_model_size(model, data_width=16, group_size=128)  # TODO: is the group_size param accurate?
    print(f"LLM backbone model size: {model_size/MiB:.2f} MiB")

    # pseudo_quantize_model_weight(model, w_bit=3, q_group_size=128)
    input_feat = get_calib_feat(model, tokenizer, processor)  # Note this is input_feat for LLM backbone and not OpenVLA
    pseudo_quantize_model_weight_auto_scale(model, w_bit=3, q_group_size=128, input_feat=input_feat)
    model_size = get_model_size(model, data_width=3, group_size=128)
    print(f"LLM backbone model size after AWQ auto scale quantization: {model_size/MiB:.2f} MiB")

    vla.language_model = model
    return vla
