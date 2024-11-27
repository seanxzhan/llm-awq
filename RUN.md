# Installation

Torch version 2.2.0, cuda 11.8:
```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Example activation script:
```bash
conda activate /envs/awq

export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME=/usr/local/cuda-11.8
export HF_HOME=/datasets/cache
```

Install openvla inside of llm-awq:
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