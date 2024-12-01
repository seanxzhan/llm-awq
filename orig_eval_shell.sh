#python -m awq.entry --model_path openvla/openvla-7b \
#    --baseline --eval_set_test\
#    --batch_size 2 \
#    --eval_root_dir eval \
#    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
#    --dataset_name bridge_orig \
#    --expname orig-1
#python -m awq.entry --model_path openvla/openvla-7b \
#    --baseline --eval_set_test\
#    --batch_size 2 \
#    --eval_root_dir eval \
#    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
#    --dataset_name bridge_orig \
#    --expname orig-2 
python -m awq.entry --model_path openvla/openvla-7b \
    --baseline --eval_set_test\
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname orig-3 
python -m awq.entry --model_path openvla/openvla-7b \
    --baseline --eval_set_test\
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname orig-4
python -m awq.entry --model_path openvla/openvla-7b \
    --baseline --eval_set_test\
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /mnt/align4_drive/rachelm8/tinyml \
    --dataset_name bridge_orig \
    --expname orig-5