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

Put the calibration dataset anywhere you want and change `dataset_dir` in `awq/utils/calib_data.py`'s `get_calib_dataset_openvla` function.

Put the evaliation dataset anywhere you want and change `data_root_dir` in the awq commands below.

Change `save_dataset_statistics` function in `openvla/prismatic/vla/datasets/rlds/utils/data_utils.py`:
```bash
def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    # out_path = run_dir / "dataset_statistics.json"
    out_path = os.path.join(run_dir, "dataset_statistics.json")
```

Commands to run experiments
```bash
# run awq
python -m awq.entry --model_path openvla/openvla-7b \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/openvla.pt \
    --calib_data openvla

# pseudo quantize
python -m awq.entry --model_path openvla/openvla-7b \
    --w_bit 4 --q_group_size 128 \
    --q_backend fake

# evaluation on bridge_orig with pretrained weights
python -m awq.entry --model_path openvla/openvla-7b \
    --baseline --cuda_no_double\
    --tasks bridge_orig \
    --w_bit 4 --q_group_size 128 \
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --expname original 

# evaluation on bridge_orig with awq pseudo quant
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks bridge_orig \
    --w_bit 4 --q_group_size 128 --cuda_no_double\
    --load_awq awq_cache/openvla.pt \
    --q_backend fake \
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --expname fake 

# generate real quantized weights
mkdir quant_cache
python -m awq.entry --model_path /PATH/TO/LLAMA3/llama3-8b \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/llama3-8b-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/llama3-8b-w4-g128-awq.pt
```

`auto_scale_block` points to `LlamaDecoderLayer` in https://vscode.dev/github/seanxzhan/llm-awq/blob/main/awq/quantize/auto_scale.py#L214

Relevant files: `entry.py`, `pre_quant.py`, `calib_data.py`, `auto_scale.py`, `quantizer.py`.