python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test \
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname salient
python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test \
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname salient-2
python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test \
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname salient-3
python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test \
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname salient-4
python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test \
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname salient-5
