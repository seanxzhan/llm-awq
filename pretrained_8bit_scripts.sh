# run awq
python -m awq.entry --model_path openvla/openvla-7b \
    --w_bit 8 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/openvla_8bit.pt \
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
    --expname naive-8bit
# evaluation on bridge_orig with pseudo linear salient quant (on test split)
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks linear_salient_eval \
    --w_bit 8 --q_group_size 128 \
    --calib_data openvla --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname salient-8bit
# evaluation on bridge_orig with awq pseudo quant (on test split)
python -m awq.entry --model_path openvla/openvla-7b \
    --tasks bridge_orig \
    --w_bit 8 --q_group_size 128 \
    --q_backend fake \
    --load_awq awq_cache/openvla_8bit.pt \
    --batch_size 1 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname awq-8bit
