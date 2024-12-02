python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets (ACC)' \
 --file_paths 'awq/summary_csvs/acc_summary_csv/orig_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/naive_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/salient_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/awq_summary-acc.csv' \
 --xlabel "Iteration" --ylabel "Accuracy"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets (ACC6)' \
 --file_paths 'awq/summary_csvs/acc6_summary_csvs/orig_summary-acc6.csv' 'awq/summary_csvs/acc6_summary_csvs/naive_summary-acc6.csv' 'awq/summary_csvs/acc6_summary_csvs/salient_summary-acc6.csv' 'awq/summary_csvs/acc6_summary_csvs/awq_summary-acc6.csv' \
 --xlabel "Iteration" --ylabel "Accuracy"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets (L1LOSSNORM)' \
 --file_paths 'awq/summary_csvs/l1lossnorm_summary_csvs/orig_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/naive_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/salient_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/awq_summary-l1lossnorm.csv' \
 --xlabel "Iteration" --ylabel "Loss"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets (L1LOSSNORM6)' \
 --file_paths 'awq/summary_csvs/l1lossnorm6_summary_csvs/orig_summary-l1lossnorm6.csv' 'awq/summary_csvs/l1lossnorm6_summary_csvs/naive_summary-l1lossnorm6.csv' 'awq/summary_csvs/l1lossnorm6_summary_csvs/salient_summary-l1lossnorm6.csv' 'awq/summary_csvs/l1lossnorm6_summary_csvs/awq_summary-l1lossnorm6.csv' \
 --xlabel "Iteration" --ylabel "Loss"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets (L2LOSSNORM)' \
 --file_paths 'awq/summary_csvs/l2lossnorm_summary_csvs/orig_summary-l2lossnorm.csv' 'awq/summary_csvs/l2lossnorm_summary_csvs/naive_summary-l2lossnorm.csv' 'awq/summary_csvs/l2lossnorm_summary_csvs/salient_summary-l2lossnorm.csv' 'awq/summary_csvs/l2lossnorm_summary_csvs/awq_summary-l2lossnorm.csv' \
 --xlabel "Iteration" --ylabel "Loss"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets (L2LOSSNORM6)' \
 --file_paths 'awq/summary_csvs/l2lossnorm6_summary_csvs/orig_summary-l2lossnorm6.csv' 'awq/summary_csvs/l2lossnorm6_summary_csvs/naive_summary-l2lossnorm6.csv' 'awq/summary_csvs/l2lossnorm6_summary_csvs/salient_summary-l2lossnorm6.csv' 'awq/summary_csvs/l2lossnorm6_summary_csvs/awq_summary-l2lossnorm6.csv' \
 --xlabel "Iteration" --ylabel "Loss"
#Finetuned
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned (ACC)' \
 --file_paths 'awq/summary_csvs/acc_summary_csv/lora-orig_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-naive_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-salient_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-awq_summary-acc.csv' \
 --xlabel "Iteration" --ylabel "Accuracy"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned(ACC6)' \
 --file_paths 'awq/summary_csvs/acc6_summary_csvs/lora-orig_summary-acc6.csv' 'awq/summary_csvs/acc6_summary_csvs/lora-naive_summary-acc6.csv' 'awq/summary_csvs/acc6_summary_csvs/lora-salient_summary-acc6.csv' 'awq/summary_csvs/acc6_summary_csvs/lora-awq_summary-acc6.csv' \
 --xlabel "Iteration" --ylabel "Accuracy"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned(L1LOSSNORM)' \
 --file_paths 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-orig_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-naive_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-salient_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-awq_summary-l1lossnorm.csv' \
 --xlabel "Iteration" --ylabel "Loss"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned (L1LOSSNORM6)' \
 --file_paths 'awq/summary_csvs/l1lossnorm6_summary_csvs/lora-orig_summary-l1lossnorm6.csv' 'awq/summary_csvs/l1lossnorm6_summary_csvs/lora-naive_summary-l1lossnorm6.csv' 'awq/summary_csvs/l1lossnorm6_summary_csvs/lora-salient_summary-l1lossnorm6.csv' 'awq/summary_csvs/l1lossnorm6_summary_csvs/lora-awq_summary-l1lossnorm6.csv' \
 --xlabel "Iteration" --ylabel "Loss"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.pyy --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned (L2LOSSNORM)' \
 --file_paths 'awq/summary_csvs/l2lossnorm_summary_csvs/lora-orig_summary-l2lossnorm.csv' 'awq/summary_csvs/l2lossnorm_summary_csvs/lora-naive_summary-l2lossnorm.csv' 'awq/summary_csvs/l2lossnorm_summary_csvs/lora-salient_summary-l2lossnorm.csv' 'awq/summary_csvs/l2lossnorm_summary_csvs/lora-awq_summary-l2lossnorm.csv' \
 --xlabel "Iteration" --ylabel "Loss"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned (L2LOSSNORM6)' \
 --file_paths 'awq/summary_csvs/l2lossnorm6_summary_csvs/lora-orig_summary-l2lossnorm6.csv' 'awq/summary_csvs/l2lossnorm6_summary_csvs/lora-naive_summary-l2lossnorm6.csv' 'awq/summary_csvs/l2lossnorm6_summary_csvs/lora-salient_summary-l2lossnorm6.csv' 'awq/summary_csvs/l2lossnorm6_summary_csvs/lora-awq_summary-l2lossnorm6.csv' \
 --xlabel "Iteration" --ylabel "Loss"
