import pandas as pd

from constants import cat2id

# Load data into DataFrame
csv_path = "results/stats/alignment/targeted_categorized_fidelity_sasrec.csv"
df = pd.read_csv(csv_path)

# Reverse the cat2id mapping to get category names
id2cat = {v: k for k, v in cat2id.items()}

# Filter categories of interest
categories = ["Drama", "Action", "Adventure", "Fantasy", "Animation", "Horror"]
filtered_df = df[
    df["gen_target_y@1"].isin([f"{'{'}{cat2id[cat]}{'}'}" for cat in categories])
]


# Generate markdown tables
md_content = ""
for cat in categories:
    cat_id = cat2id[cat]
    print(f"[DEBUG], catid is", cat_id)
    row = filtered_df[filtered_df["gen_target_y@1"] == f"{'{'}{cat_id}{'}'}"]
    if not row.empty:
        md_content += f"| {cat} Category | fidelity@1 | fidelity@5 | fidelity@10 | fidelity@20 |\n"
        md_content += (
            "|---------------|------------|------------|-------------|-------------|\n"
        )
        md_content += f"| GENE          | {row['fidelity_gen_score@1'].values[0]*100:.2f}% | {row['fidelity_gen_score@5'].values[0]*100:.2f}% | {row['fidelity_gen_score@10'].values[0]*100:.2f}% | {row['fidelity_gen_score@20'].values[0]*100:.2f}% |\n"
        md_content += f"| PACE          | {row['fidelity_score@1'].values[0]*100:.2f}% | {row['fidelity_score@5'].values[0]*100:.2f}% | {row['fidelity_score@10'].values[0]*100:.2f}% | {row['fidelity_score@20'].values[0]*100:.2f}% |\n\n"

# Save to markdown file
file_path = "./fidelity_score_table.md"
with open(file_path, "w") as f:
    f.write(md_content)
