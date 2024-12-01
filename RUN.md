# Installation

Follow installation steps 1, 2, 3 from llm-awq's README.

Install torch version 2.2.0, cuda 11.8 for the conda environment:
```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Example activation script:
```bash
conda activate /envs/awq

export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME=/usr/local/cuda-11.8
# specify where the hugging face openvla model is cached
export HF_HOME=/datasets/cache
```

Install openvla inside of llm-awq:
```bash
git clone https://github.com/openvla/openvla.git
cd openvla 
pip install -e .
```

Put the calibration dataset (`openvla-7b+calib_feat+b1--original`) anywhere you want and change `dataset_dir` in `awq/utils/calib_data.py`'s `get_calib_dataset_openvla` function.

Put the evaliation dataset (`bridge_orig`) anywhere you want and change `data_root_dir` in the awq commands below.

Change `save_dataset_statistics` function in `openvla/prismatic/vla/datasets/rlds/utils/data_utils.py` to avoid runtime error:
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
    --cali

# run awq on lora
python -m awq.entry --model_path openvla/openvla-7b \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/openvla-lora.pt \
    --cali

# Pretrained weights
python -m awq.entry --model_path openvla/openvla-7b \
    --baseline \
    --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --expname orig-train-no-last-bit

# Pretrained weights
python -m awq.entry --model_path openvla/openvla-7b \
    --baseline --eval_set_test\
    --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --expname orig-test-no-last-bit

# Pretrained weights with just linear layers pseudo quantized
python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test\
    --tasks bridge_orig \
    --w_bit 4 --q_group_size 128 \
    --q_backend fake \
    --dump_fake saved_models/naive.pt \
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --expname naive 

# evaluation on bridge_orig with pseudo linear salient quant (on test split)
python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test \
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --expname salient

# evaluation on bridge_orig with awq pseudo quant (on test split)
python -m awq.entry --model_path openvla/openvla-7b \
     --eval_set_test \
    --tasks bridge_orig \
    --w_bit 4 --q_group_size 128 \
    --q_backend fake \
    --dump_fake saved_models/awq.pt \
    --load_awq awq_cache/openvla.pt \
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --expname awq 

# Finetuned weights
python -m awq.entry --model_path openvla/openvla-7b \
    --baseline \
    --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --lora_pt /sota/openvla/finetuned.pt \
    --expname lora-orig-train-no-last-bit

# Finetuned weights
python -m awq.entry --model_path openvla/openvla-7b \
    --baseline --eval_set_test\
    --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --lora_pt /sota/openvla/finetuned.pt \
    --expname lora-orig-test-no-last-bit


python -m awq.entry --model_path openvla/openvla-7b \
     --eval_set_test \
    --tasks bridge_orig \
    --w_bit 4 --q_group_size 128 \
    --q_backend fake \
    --dump_fake saved_models/awq.pt \
    --load_awq awq_cache/openvla.pt \
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --expname lora-awq-train-no-last-bit

# generate real quantized weights
mkdir quant_cache
python -m awq.entry --model_path openvla/openvla-7b \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/openvla.pt \
    --q_backend real --dump_quant quant_cache/openvla-awq.pt

# evaluation on brige_orig with awq real quant (on test split)
# not working right now due to bfloat16 vs. float16
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks bridge_orig \
    --w_bit 4 --q_group_size 128  --eval_set_test\
    --load_quant quant_cache/openvla-awq-v2.pt \
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --expname real

# print model statistics
pip install torchprofile
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks model_statistics \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/openvla.pt
```

`auto_scale_block` points to `LlamaDecoderLayer` in https://vscode.dev/github/seanxzhan/llm-awq/blob/main/awq/quantize/auto_scale.py#L214. We can modify the function to try targetting different layers to resize.

Relevant files for us: `entry.py`, `pre_quant.py`, `calib_data.py`, `auto_scale.py`, `quantizer.py`.