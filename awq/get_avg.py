import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# plt_name = 'plot-awq-5-trials'
# file_paths = [
#     'awq/summary_csvs/acc_summary_csv/awq-1_summary.csv',
#     'awq/summary_csvs/acc_summary_csv/awq-2_summary.csv',
#     'awq/summary_csvs/acc_summary_csv/awq-3_summary.csv',
#     'awq/summary_csvs/acc_summary_csv/awq-4_summary.csv',
#     'awq/summary_csvs/acc_summary_csv/awq-5_summary.csv',
# ]
# plt_name = 'plot-orig-no-last-bit-5-trials'
# file_paths = [
#     'awq/summary_csvs/acc_summary_csv/orig-no-last-bit-1_summary.csv',
#     'awq/summary_csvs/acc_summary_csv/orig-no-last-bit-2_summary.csv',
#     'awq/summary_csvs/acc_summary_csv/orig-no-last-bit-3_summary.csv',
#     'awq/summary_csvs/acc_summary_csv/orig-no-last-bit-4_summary.csv',
#     'awq/summary_csvs/acc_summary_csv/orig-no-last-bit-5_summary.csv',
# ]
# plt_name = 'plot-all-quantize-trials-5'
# file_paths = [
#     'awq/summary_csvs/acc_summary_csv/orig-5_summary.csv',
#     'awq/summary_csvs/acc_summary_csv/naive-5_summary.csv',
#     'awq/summary_csvs/acc_summary_csv/salient-5_summary.csv',
#     'awq/summary_csvs/acc_summary_csv/awq-5_summary.csv',
# ]

plt_name = 'plot-pretrained-weights-all-expts-train-100-datapoints'
file_paths = [
    'awq/summary_csvs/tested_on_train_100_datapoints/orig-train_summary.csv',
    'awq/summary_csvs/tested_on_train_100_datapoints/naive-train_summary.csv',
    'awq/summary_csvs/tested_on_train_100_datapoints/salient-train_summary.csv',
    'awq/summary_csvs/tested_on_train_100_datapoints/awq-train_summary.csv',
]

# colors = ['green', 'blue', 'red', 'pink', 'orange']
# labels = ['awq-1', 'awq-2', 'awq-3', 'awq-4', 'awq-5']
colors = ['green', 'blue', 'red', 'pink']
labels = ['orig', 'naive', 'awq', 'salient']

# Load data into a list of dataframes
data_frames = [pd.read_csv(file_path) for file_path in file_paths]

print(list(zip(labels, [df['Value'].mean() for df in data_frames])))

# Apply the exponential moving average (EWMA) smoothing
TSBOARD_SMOOTHING = 0.85
smoothed_data_frames = [df.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean() for df in data_frames]

# Plot the original and smoothed values for each dataset
plt.figure(figsize=(6, 4))

for i, df in enumerate(data_frames):
    plt.plot(df['Step'], df['Value'], alpha=0.3, c=colors[i])
    plt.plot(df['Step'], smoothed_data_frames[i]['Value'], c=colors[i], label=f'{labels[i]}')

plt.xlabel('Iteration')
plt.ylabel('Average')
plt.title(plt_name)
plt.grid(alpha=0.3)
plt.legend()

plt.savefig(f'awq/plots/{plt_name}.png', bbox_inches='tight', pad_inches=0, dpi=150, 
            transparent=False)
print(f'file saved to {plt_name}.png')

# Open the file in write mode ('w')
with open(f"awq/plots/{plt_name}-averages.txt", "w") as file:
    # Write content to the file
    means = list(zip(labels, [df['Value'].mean() for df in data_frames]))
    file.write(str(means))
    mean_values = [mean for _, mean in means]

    # Calculate the mean of the means
    mean_of_means = np.mean(mean_values)
    print(mean_of_means)
    file.write(f"\nMean of means: {mean_of_means}")

print("Content written to file.")
