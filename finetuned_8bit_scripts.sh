# run awq on lora
python -m awq.entry --model_path openvla/openvla-7b \
    --w_bit 8 --q_group_size 128 \
    --lora_pt /mnt/align4_drive/rachelm8/llm-awq/finetuned/finetuned.pt \
    --run_awq --dump_awq awq_cache/openvla-lora_8bit.pt \
    --calib_data openvla
# Pretrained weights with just linear layers pseudo quantized
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks bridge_orig \
    --w_bit 8 --q_group_size 128 \
    --q_backend fake \
    --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --lora_pt /mnt/align4_drive/rachelm8/llm-awq/finetuned/finetuned.pt \
    --expname lora-naive-8bit
# evaluation on bridge_orig with pseudo linear salient quant (on test split)
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks linear_salient_eval \
    --w_bit 8 --q_group_size 128 \
    --calib_data openvla --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --lora_pt /mnt/align4_drive/rachelm8/llm-awq/finetuned/finetuned.pt \
    --expname lora-salient-8bit
# evaluation on bridge_orig with awq pseudo quant on finetuned weights
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks bridge_orig \
    --w_bit 8 --q_group_size 128 \
    --q_backend fake \
    --load_awq awq_cache/openvla-lora_8bit.pt \
    --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --lora_pt /mnt/align4_drive/rachelm8/llm-awq/finetuned/finetuned.pt \
    --expname lora-awq-8bit
