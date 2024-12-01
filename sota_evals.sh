#orig lora
python -m awq.entry --model_path openvla/openvla-7b \
   --baseline --eval_set_test\
   --batch_size 2 \
   --eval_root_dir eval \
   --data_root_dir /datasets \ 
   --dataset_name bridge_orig \
   --lora_pt /sota/openvla/finetuned.pt \
   --expname lora-orig
#naive lora
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
    --lora_pt /sota/openvla/finetuned.pt \
    --expname lora-naive
#salient lora
python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test \
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /datasets \ 
    --dataset_name bridge_orig \
    --lora_pt /sota/openvla/finetuned.pt \
    --expname lora-salient
#awq lora
python -m awq.entry --model_path openvla/openvla-7b \
     --eval_set_test \
    --tasks bridge_orig \
    --w_bit 4 --q_group_size 128 \
    --q_backend fake \
    --dump_fake saved_models/awq.pt \
    --load_awq awq_cache/openvla.pt \
    --batch_size 2 \
    --data_root_dir /datasets \ 
    --dataset_name bridge_orig \
    --lora_pt /sota/openvla/finetuned.pt \
    --expname lora-awq
#orig-no-last-bit lora
python -m awq.entry --model_path openvla/openvla-7b \
   --baseline --eval_set_test --no_last_bit_eval \
   --batch_size 2 \
   --eval_root_dir eval \
   --data_root_dir /datasets \ 
   --dataset_name bridge_orig \
   --lora_pt /sota/openvla/finetuned.pt \
   --expname lora-orig
#naive-no-last-bit lora
python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test --no_last_bit_eval \
    --tasks bridge_orig \
    --w_bit 4 --q_group_size 128 \
    --q_backend fake \
    --dump_fake saved_models/naive.pt \
    --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /datasets \ 
    --dataset_name bridge_orig \
    --lora_pt /sota/openvla/finetuned.pt \
    --expname lora-naive
#salient-no-last-bit lora
python -m awq.entry --model_path openvla/openvla-7b \
    --eval_set_test --no_last_bit_eval \
    --tasks linear_salient_eval \
    --w_bit 4 --q_group_size 128 \
    --calib_data openvla --batch_size 2 \
    --eval_root_dir eval \
    --data_root_dir /datasets \ 
    --dataset_name bridge_orig \
    --lora_pt /sota/openvla/finetuned.pt \
    --expname lora-salient
#awq-no-last-bit lora
python -m awq.entry --model_path openvla/openvla-7b \
     --eval_set_test --no_last_bit_eval \
    --tasks bridge_orig \
    --w_bit 4 --q_group_size 128 \
    --q_backend fake \
    --dump_fake saved_models/awq.pt \
    --load_awq awq_cache/openvla.pt \
    --batch_size 2 \
    --data_root_dir /datasets \ 
    --dataset_name bridge_orig \
    --lora_pt /sota/openvla/finetuned.pt \
    --expname lora-awq