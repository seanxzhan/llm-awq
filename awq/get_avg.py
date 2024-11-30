import pandas as pd
import matplotlib.pyplot as plt

plt_name = 'plot-orig-naive-awq'
file_paths = [
    # 'awq/summary_csvs/orig-train-vs.csv',
    # 'awq/summary_csvs/lora-train-vs.csv',
    # 'awq/summary_csvs/orig_actionl1.csv',
    # 'awq/summary_csvs/naive_actionl1.csv',
    # 'awq/summary_csvs/awq_actionl1.csv',
    'awq/summary_csvs/orig.csv',
    'awq/summary_csvs/naive.csv',
    'awq/summary_csvs/awq.csv',
]
colors = ['green', 'blue', 'red']
labels = ['orig', 'naive', 'awq']

# Load data into a list of dataframes
data_frames = [pd.read_csv(file_path) for file_path in file_paths]

print([df['Value'].mean() for df in data_frames])

# Apply the exponential moving average (EWMA) smoothing
TSBOARD_SMOOTHING = 0.85
smoothed_data_frames = [df.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean() for df in data_frames]

# Plot the original and smoothed values for each dataset
plt.figure(figsize=(6, 4))

for i, df in enumerate(data_frames):
    plt.plot(df['Step'], df['Value'], alpha=0.3, c=colors[i])
    plt.plot(df['Step'], smoothed_data_frames[i]['Value'], c=colors[i], label=f'{labels[i]}')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title(plt_name)
plt.grid(alpha=0.3)
plt.legend()

plt.savefig(f'{plt_name}.png', bbox_inches='tight', pad_inches=0, dpi=150, 
            transparent=False)
print(f'file saved to {plt_name}.png')
