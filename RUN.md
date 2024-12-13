# Installation

Follow installation steps 1, 2, 3 from llm-awq's README.

Install torch version 2.2.0, cuda 11.8 for the conda environment:
```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Example activation script:
```bash
conda activate awq

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

# Datasets

The three datasets mentioned below are in this [Goolge Drive folder](https://drive.google.com/drive/u/1/folders/1hzRDgc7UNzLYUd7zWuKJO9nde21yw026).

Put the calibration dataset (`multimodal_embeddings`) anywhere you want and change `dataset_dir` in `awq/utils/calib_data.py`'s `get_calib_dataset_openvla` function. If you'd like to try out calibrating AWQ with just the language embeddings, use `language_embeddings`.

Put the evaliation dataset (`bridge_orig`) anywhere you want and change `data_root_dir` in the awq commands below.

# OpenVLA Code Modification (Optional)

Change `save_dataset_statistics` function in `openvla/prismatic/vla/datasets/rlds/utils/data_utils.py` to avoid runtime error if needed:
```bash
def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    # out_path = run_dir / "dataset_statistics.json"
    out_path = os.path.join(run_dir, "dataset_statistics.json")
```

# Experiments

Commands to run experiments

(Add ```--no_last_bit_eval``` for evaluating for 6 bits)
```bash
# run awq
python -m awq.entry --model_path openvla/openvla-7b \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/openvla.pt \
    --calib_data openvla

# run awq on lora
python -m awq.entry --model_path openvla/openvla-7b \
    --w_bit 4 --q_group_size 128 \
    --lora_pt /sota/openvla/finetuned.pt \
    --run_awq --dump_awq awq_cache/openvla-lora.pt \
    --calib_data openvla

# evaluate on OpenVLA pretrained weights
python -m awq.entry --model_path openvla/openvla-7b \
    --baseline \
    --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --expname orig

# evaluate on OpenVLA pretrained weights with all linear layers quantized
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks bridge_orig \
    --w_bit 4 --q_group_size 128 \
    --q_backend fake \
    --dump_fake saved_models/naive.pt \
    --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --expname naive

# evaluate on OpenVLA pretrained weights with salient quantization (keep 1% of salient at bfloat16)
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --expname salient

# evaluate on OpenVLA pretrained weights with awq quantization (pseudo weights)
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks bridge_orig \
    --w_bit 4 --q_group_size 128 \
    --q_backend fake \
    --dump_fake saved_models/awq.pt \
    --load_awq awq_cache/openvla.pt \
    --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --expname awq

# evaluate on lora finetuned weights
python -m awq.entry --model_path openvla/openvla-7b \
    --baseline \
    --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --lora_pt /sota/openvla/finetuned.pt \
    --expname lora-orig

# evaluate on lora finetuned weights with all linear layers quantized
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks bridge_orig \
    --w_bit 4 --q_group_size 128 \
    --q_backend fake \
    --dump_fake saved_models/lora-naive.pt \
    --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --lora_pt /sota/openvla/finetuned.pt \
    --expname lora-naive

# evaluate on lora finetuned weights with salient quantization (keep 1% of salient at bfloat16)
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --lora_pt /sota/openvla/finetuned.pt \
    --expname lora-salient

# evaluate on lora finetuned weights with awq quantization (pseudo weights)
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks bridge_orig \
    --w_bit 4 --q_group_size 128 \
    --q_backend fake \
    --dump_fake saved_models/lora-awq.pt \
    --load_awq awq_cache/openvla-lora.pt \
    --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /datasets \
    --dataset_name bridge_orig \
    --lora_pt /sota/openvla/finetuned.pt \
    --expname lora-awq

# generate awq real quantized weights
mkdir quant_cache
python -m awq.entry --model_path openvla/openvla-7b \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/openvla.pt \
    --q_backend real --dump_quant quant_cache/openvla-awq.pt

# evaluation on awq real quantized weights
# not working right now due to bfloat16 vs. float16 error in c backend
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
