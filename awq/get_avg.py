import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

plt_name = 'Step vs Value With Smoothing for Multiple Datasets (ACC)'
file_paths = [
    'awq/summary_csvs/acc_summary_csv/orig_summary-acc.csv',
    'awq/summary_csvs/acc_summary_csv/naive_summary-acc.csv',
    'awq/summary_csvs/acc_summary_csv/salient_summary-acc.csv',
    'awq/summary_csvs/acc_summary_csv/awq_summary-acc.csv',
]
def parse_args():
    parser = argparse.ArgumentParser(description="Process plot parameters and file paths.")
    
    # Argument for plot name
    parser.add_argument('--plt_name', type=str, required=True, help="The name of the plot")
    
    # Argument for file paths (can accept a list of strings)
    parser.add_argument('--file_paths', type=str, nargs='+', required=True, help="List of file paths to the CSV data")
    
    # Arguments for the x and y labels
    parser.add_argument('--xlabel', type=str, default='Iteration', help="Label for the x-axis")
    parser.add_argument('--ylabel', type=str, default='Average', help="Label for the y-axis")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Colors and labels for different datasets
    colors = ['green', 'blue', 'red', 'pink']
    labels = ['orig', 'naive', 'salient', 'awq']
    
    # Load data into a list of dataframes
    data_frames = [pd.read_csv(file_path) for file_path in args.file_paths]

    # Print means of the 'Value' column from each dataframe
    print(list(zip(labels, [df['Value'].mean() for df in data_frames])))

    # Apply the exponential moving average (EWMA) smoothing
    TSBOARD_SMOOTHING = 0.85
    smoothed_data_frames = [df.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean() for df in data_frames]

    # Plot the original and smoothed values for each dataset
    plt.figure(figsize=(6, 4))

    for i, df in enumerate(data_frames):
        plt.plot(df['Step'], df['Value'], alpha=0.3, c=colors[i])
        plt.plot(df['Step'], smoothed_data_frames[i]['Value'], c=colors[i], label=f'{labels[i]}')

    # Set plot labels and title
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(args.plt_name)
    plt.grid(alpha=0.3)
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(f'awq/plots/{args.plt_name}.png', bbox_inches='tight', pad_inches=0, dpi=150, transparent=False)
    print(f'File saved to {args.plt_name}.png')

    # Open the file in write mode ('w') and save the means
    with open(f"awq/plots/{args.plt_name}-averages.txt", "w") as file:
        # Calculate and write means of the 'Value' column
        means = list(zip(labels, [df['Value'].mean() for df in data_frames]))
        file.write(str(means))
        #mean_values = [mean for _, mean in means]

        # Calculate the mean of the means
        #mean_of_means = np.mean(mean_values)
        #print(mean_of_means)
        #file.write(f"\nMean of means: {mean_of_means}")

    print("Content written to file.")

if __name__ == "__main__":
    main()

""" colors = ['green', 'blue', 'red', 'pink']
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
 """