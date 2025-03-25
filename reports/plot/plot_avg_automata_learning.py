import matplotlib.pyplot as plt
import numpy as np

# Data from the table
settings = [
    "Targeted-Categorized", "Targeted-Uncategorized", 
    "Untargeted-Uncategorized", "Untargeted-Categorized"
]

precision = [0.7950, 0.9111, 0.8149, 0.7895]
accuracy = [0.5507, 0.6595, 0.5149, 0.5987]
recall = [0.3086, 0.6045, 0.2158, 0.2887]

x = np.arange(len(settings))  # X locations for the groups
width = 0.2  # Width of the bars

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
bars1 = ax.bar(x - 1.5*width, precision, width, label='Precision', color='blue')
bars2 = ax.bar(x - 0.5*width, accuracy, width, label='Accuracy', color='lightblue')
bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='red')

# Function to add text labels on bars
def add_value(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

def add_method_name(bars, method_name, color="white"):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height / 2, method_name, 
                ha='center', va='center', fontsize=10, fontweight='bold', color=color)

# Add labels to each bar
add_value(bars1)
add_value(bars2)
add_value(bars3)

add_method_name(bars1, "Prec.")
add_method_name(bars2, "Acc.")
add_method_name(bars3, "Rec.")

    # Labels and title
ax.set_xlabel("Setting")
ax.set_ylabel("Evalution Metrics")
ax.set_title("Automata Learning Evalaution Across Different Settings")
ax.set_xticks(x)
ax.set_xticklabels(settings, rotation=00, ha="center")
ax.legend()

# Show plot
plt.tight_layout()
plt.show()

