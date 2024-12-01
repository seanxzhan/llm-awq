python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test --no_last_bit_eval\
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname salient-no-last-bit-1
python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test --no_last_bit_eval\
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname salient-no-last-bit-2
python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test --no_last_bit_eval\
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname salient-no-last-bit-3
python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test --no_last_bit_eval\
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname salient-no-last-bit-4
python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test --no_last_bit_eval\
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname salient-no-last-bit-5