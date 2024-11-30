import pandas as pd
import matplotlib.pyplot as plt

plt_fn = 'plot-pseudo'
file_paths = [
    './summary_csvs/openvla-7b+bridge_orig+b2--original_summary.csv',
    './summary_csvs/openvla-7b+bridge_orig+b2--pseudo_summary.csv',
    './summary_csvs/openvla-7b+bridge_orig+b2--pseudo_salient_summary.csv',
    # 'ckpt/openvla-7b+bridge_orig+b2--awq_scale_summary.csv',
    # 'ckpt/openvla-7b+bridge_orig+b2--awq_scale_2_summary.csv',
    # 'ckpt/openvla-7b+bridge_orig+b2--awq_scale_3_summary.csv',
]
colors = ['green', 'blue', 'red']
labels = ['original', 'pseudo', 'pseudo salient']

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
plt.title('Step vs Value with Smoothing for Multiple Datasets')
plt.grid(alpha=0.3)
plt.legend()

plt.savefig(f'{plt_fn}.png', bbox_inches='tight', pad_inches=0, dpi=150, 
            transparent=False)
print(f'file saved to {plt_fn}.png')

# cds = [
#     0.0006709914837992497,
#     0.002600045045553819,
#     0.0005373566555956704,
#     0.0006803622840710696,
#     0.0006286145327900595,
#     0.0006065586382294539,
#     0.0009073988923759162,
#     0.0007341729792768458,
#     0.0006351928217864726,
#     0.0006969988725196135,
# ]

# import numpy as np
# print(np.mean(cds))

# awq_scale: wrong
# awq scale 2: wrt to not just text input_ids, w img feature, just scale attn
# awq scale 3: wrt to not just text input_ids, w img feature, scale attn and fc
