python -m awq.entry --model_path openvla/openvla-7b \
    --baseline --eval_set_test --no_last_bit_eval\
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname orig-no-last-bit-1
python -m awq.entry --model_path openvla/openvla-7b \
    --baseline --eval_set_test --no_last_bit_eval\
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname orig-no-last-bit-2
    python -m awq.entry --model_path openvla/openvla-7b \
    --baseline --eval_set_test --no_last_bit_eval\
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname orig-no-last-bit-3
python -m awq.entry --model_path openvla/openvla-7b \
    --baseline --eval_set_test --no_last_bit_eval\
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname orig-no-last-bit-4
python -m awq.entry --model_path openvla/openvla-7b \
    --baseline --eval_set_test --no_last_bit_eval\
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname orig-no-last-bit-5