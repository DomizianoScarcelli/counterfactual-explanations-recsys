import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('results/stats/model_sensitivity_category_positions.csv')

# Filter relevant columns
x = ["position"]
y = ['all_changes', 'any_changes', 'jaccards']
columns_to_use =  x + y
data = data[columns_to_use]

# Group data by 'k'
k_values = None
if "k" in data:
    k_values = data['k'].unique()

if k_values:
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
else:
    subset = data
    # Plot metrics for different positions
    plt.figure(figsize=(10, 6))
    
    plt.plot(subset['position'], subset['all_changes'], label='% sequences all categories changed',  marker='o')
    plt.plot(subset['position'], subset['any_changes'], label='% sequences at least one category changed', marker="s")
    plt.plot(subset['position'], 100 - subset['jaccards'], label='Jaccard Distance', marker="^")

    plt.title(f'Metrics for categorized (k = 1)')
    plt.xlabel('Changed element at index')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"plot/figs/model_sensitivity_category.png")
