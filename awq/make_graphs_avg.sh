python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets (ACC)' \
 --file_paths 'awq/summary_csvs/acc_summary_csv/orig_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/naive_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/salient_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/awq_summary-acc.csv' \
 --xlabel "Iteration" --ylabel "Accuracy"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets (L1LOSSNORM)' \
 --file_paths 'awq/summary_csvs/l1lossnorm_summary_csvs/orig_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/naive_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/salient_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/awq_summary-l1lossnorm.csv' \
 --xlabel "Iteration" --ylabel "Loss"
#2bit
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets 2bit (ACC)' \
 --file_paths 'awq/summary_csvs/acc_summary_csv/orig_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/naive-2bit_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/salient-2bit_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/awq-2bit_summary-acc.csv' \
 --xlabel "Iteration" --ylabel "Accuracy"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets 2bit (L1LOSSNORM)' \
 --file_paths 'awq/summary_csvs/l1lossnorm_summary_csvs/orig_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/naive-2bit_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/salient-2bit_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/awq-2bit_summary-l1lossnorm.csv' \
 --xlabel "Iteration" --ylabel "Loss"
#3bit
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets 3bit (ACC)' \
 --file_paths 'awq/summary_csvs/acc_summary_csv/orig_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/naive-3bit_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/salient-3bit_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/awq-3bit_summary-acc.csv' \
 --xlabel "Iteration" --ylabel "Accuracy"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets 3bit (L1LOSSNORM)' \
 --file_paths 'awq/summary_csvs/l1lossnorm_summary_csvs/orig_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/naive-3bit_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/salient-3bit_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/awq-3bit_summary-l1lossnorm.csv' \
 --xlabel "Iteration" --ylabel "Loss"
#8bit
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets 8bit (ACC)' \
 --file_paths 'awq/summary_csvs/acc_summary_csv/orig_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/naive_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/salient_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/awq_summary-acc.csv' \
 --xlabel "Iteration" --ylabel "Accuracy"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets 8bit (L1LOSSNORM)' \
 --file_paths 'awq/summary_csvs/l1lossnorm_summary_csvs/orig_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/naive-8bit_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/salient-8bit_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/awq-8bit_summary-l1lossnorm.csv' \
 --xlabel "Iteration" --ylabel "Loss"
#Finetuned
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned (ACC)' \
 --file_paths 'awq/summary_csvs/acc_summary_csv/lora-orig_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-naive_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-salient_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-awq_summary-acc.csv' \
 --xlabel "Iteration" --ylabel "Accuracy"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned(L1LOSSNORM)' \
 --file_paths 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-orig_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-naive_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-salient_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-awq_summary-l1lossnorm.csv' \
 --xlabel "Iteration" --ylabel "Loss"
#2bit
 python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned 2bit (ACC)' \
 --file_paths 'awq/summary_csvs/acc_summary_csv/lora-orig_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-naive-2bit_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-salient-2bit_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-awq-2bit_summary-acc.csv' \
 --xlabel "Iteration" --ylabel "Accuracy"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned 2bit (L1LOSSNORM)' \
 --file_paths 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-orig_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-naive-2bit_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-salient-2bit_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-awq-2bit_summary-l1lossnorm.csv' \
 --xlabel "Iteration" --ylabel "Loss"
#3bit
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned 3bit (ACC)' \
 --file_paths 'awq/summary_csvs/acc_summary_csv/lora-orig_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-naive-3bit_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-salient-3bit_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-awq-3bit_summary-acc.csv' \
 --xlabel "Iteration" --ylabel "Accuracy"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned 3bit (L1LOSSNORM)' \
 --file_paths 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-orig_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-naive-3bit_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-salient-3bit_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-awq-3bit_summary-l1lossnorm.csv' \
 --xlabel "Iteration" --ylabel "Loss"
#8bit
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned 8bit (ACC)' \
 --file_paths 'awq/summary_csvs/acc_summary_csv/lora-orig_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-naive-8bit_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-salient-8bit_summary-acc.csv' 'awq/summary_csvs/acc_summary_csv/lora-awq-8bit_summary-acc.csv' \
 --xlabel "Iteration" --ylabel "Accuracy"
python /mnt/align4_drive/rachelm8/llm-awq/awq/get_avg.py --plt_name 'Step vs Value With Smoothing for Multiple Datasets Finetuned 8bit (L1LOSSNORM)' \
 --file_paths 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-orig_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-naive-8bit_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-salient-8bit_summary-l1lossnorm.csv' 'awq/summary_csvs/l1lossnorm_summary_csvs/lora-awq-8bit_summary-l1lossnorm.csv' \
 --xlabel "Iteration" --ylabel "Loss"