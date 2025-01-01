import matplotlib.pyplot as plt
import pandas as pd

from cli import CLI

# Example data structure
data = CLI().stats(
    what="alignment", log_path="results/evaluate_alignment_new_conf.csv"
)
data = [
    {
        "split": x["split"],
        "good": x["status"].get("good", 0),
        "bad": x["status"].get("bad", 0) + x["status"].get("CounterfactualNotFound", 0),
    }
    for x in data
]
data = [x for x in data if x["good"] + x["bad"] >= 10]

# Convert to DataFrame
df = pd.DataFrame(data)

# Number of splits
n_splits = len(df)

# Create a figure with multiple subplots
fig, axes = plt.subplots(1, n_splits, figsize=(n_splits * 4, 5), sharey=False)

# Ensure axes is iterable (for single subplot cases)
if n_splits == 1:
    axes = [axes]

# Loop through splits to create individual plots
for i, (index, row) in enumerate(df.iterrows()):
    ax = axes[i]  # Current axis
    split = row["split"]
    good = row["good"]
    bad = row["bad"]
    total = good + bad

    # Calculate percentages
    good_pct = (good / total) * 100 if total > 0 else 0
    bad_pct = (bad / total) * 100 if total > 0 else 0

    # Plot stacked bar
    ax.bar(["Counts"], [good], color="green", label="Good", alpha=0.8)
    ax.bar(["Counts"], [bad], bottom=[good], color="red", label="Bad", alpha=0.8)

    # Add percentages inside bars
    ax.text(
        0,  # Horizontal position (center of the bar stack)
        good / 2,  # Vertical position (middle of the "Good" bar)
        f"{good_pct:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontsize=10,
    )
    if bad > 0:  # Only add text for bad if it exists
        ax.text(
            0,  # Horizontal position (center of the bar stack)
            good + bad / 2,  # Vertical position (middle of the "Bad" bar)
            f"{bad_pct:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )

    # Add title for the subplot
    ax.set_title(f"Split {split}")
    # Modify the x-axis label to include the total count
    ax.set_xlabel(f"Counts ({total})")

# Add a shared legend
fig.legend(
    ["Good", "Bad"], loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.05)
)

# Adjust layout for better spacing
plt.tight_layout()

# Show plot
plt.savefig("plot/figs/different_splits_run.png")
