# 
```bash
git clone https://github.com/openvla/openvla.git
cd openvla 
pip install -e .
```

```bash
# run awq
python -m awq.entry --model_path openvla/openvla-7b \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/openvla.pt \
    --calib_data openvla

# pseudo
python -m awq.entry --model_path openvla/openvla-7b \
    --w_bit 4 --q_group_size 128 \
    --q_backend fake
```

`auto_scale_block` points to `LlamaDecoderLayer` in https://vscode.dev/github/seanxzhan/llm-awq/blob/main/awq/quantize/auto_scale.py#L214

Relevant files: `entry.py`, `pre_quant.py`, `calib_data.py`, `auto_scale.py`, `quantizer.py`.