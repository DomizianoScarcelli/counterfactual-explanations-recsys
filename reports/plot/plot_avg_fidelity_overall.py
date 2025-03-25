import matplotlib.pyplot as plt
import numpy as np

# Data from the table
settings = [
    "Targeted-Categorized", "Targeted-Uncategorized", 
    "Untargeted-Uncategorized", "Untargeted-Categorized"
]

fidelity_gene_1 = [1.13, 1.99, 1.00, 1.00]
# fidelity_gene_5 = [1.12, 1.99, 1.00, 1.00]
fidelity_pace_1 = [0.61, 0.95, 1.00, 1.00]
# fidelity_pace_5 = [0.60, 0.95, 1.00, 1.00]

x = np.arange(len(settings))  # X locations for the groups
width = 0.2  # Width of the bars

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
bars1 = ax.bar(x - 1.5*width, fidelity_gene_1, width, label='GENE distance', color='blue')
bars2 = ax.bar(x - 0.5*width, fidelity_pace_1, width, label='PACE distance', color='red')
# bars3 = ax.bar(x + 0.5*width, fidelity_pace_1, width, label='PACE distance@1', color='red')
# bars4 = ax.bar(x + 1.5*width, fidelity_pace_5, width, label='PACE distance@5', color='orange')

# Function to add text labels on bars
def add_value(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

def add_method_name(bars, method_name, color="white"):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height / 2, method_name, 
                ha='center', va='center', fontsize=10, fontweight='bold', color=color)

# Add labels to each bar
add_value(bars1)
add_value(bars2)
# add_value(bars3)
# add_value(bars4)

add_method_name(bars1, "GENE")
add_method_name(bars2, "PACE")
# add_method_name(bars3, "PACE")
# add_method_name(bars4, "PACE")

    # Labels and title
ax.set_xlabel("Setting")
ax.set_ylabel("Edit Distance")
ax.set_title("Edit Distance for GENE and PACE Across Different Settings")
ax.set_xticks(x)
ax.set_xticklabels(settings, rotation=00, ha="center")
ax.legend()

# Show plot
plt.tight_layout()
plt.savefig("distance.png")

