import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset (replace 'data.csv' with your actual file path)
data = pd.read_csv('results/model_sensitivity.csv')

# Filter relevant columns
columns_to_use = ['k', 'position', 'mean_precision', 'mean_ndcgs', 'mean_jaccard']
data = data[columns_to_use]

# Group data by 'k'
k_values = data['k'].unique()

for k in k_values:
    # Filter data for the current k value
    subset = data[data['k'] == k]
    # max_position = data['position'].max()
    # data['position'] = max_position - data['position']

    # Plot metrics for different positions
    plt.figure(figsize=(10, 6))

    plt.plot(subset['position'], subset['mean_precision'], label='Mean Precision', marker='o')
    plt.plot(subset['position'], subset['mean_ndcgs'], label='Mean NDCG', marker='s')
    plt.plot(subset['position'], subset['mean_jaccard'], label='Mean Jaccard', marker='^')

    plt.title(f'Metrics for k = {k}')
    plt.xlabel('Changed element at index')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"plot/figs/model_sensitivity_at_{k}.png")

