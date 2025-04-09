import pandas as pd

# Create a DataFrame from the CSV data
df_base = pd.read_csv("reports/csv/baseline_fidelities.csv")
df_fidel = pd.read_csv("reports/csv/fidelities.csv")

# Define the LaTeX table template
latex_template = """
\\begin{table*}[h]
\\begin{tabular}{c||c|c|c|c||c|c|c|c||c|c|c|c}
\\toprule
 & \\multicolumn{4}{c||}{Steam} & \\multicolumn{4}{c||}{ML-1M} & \\multicolumn{4}{c}{ML-100K} \\\\
\\midrule
 & un\\_un & tag\\_un & un\\_cat & tag\\_cat & un\\_un & tag\\_un & un\\_cat & tag\\_cat & un\\_un & tag\\_un & un\\_cat & tag\\_cat \\\\
 \\midrule
"""

for model in ["SASRec", "BERT4Rec", "GRU4Rec"]:
    row = f"{model}"
    for dataset in ["steam", "ml-1m", "ml-100k"]:
        for setting in ["un_un", "tag_un", "un_cat", "tag_cat"]:
            row += f" & {{{model.lower()}_gece_{dataset}_{setting}}}"
    latex_template += row + "\n"

latex_template += "\\midrule\n"

for model in ["SASRec", "BERT4Rec", "GRU4Rec"]:
    row = f"{model} (random)"
    for dataset in ["steam", "ml-1m", "ml-100k"]:
        for setting in ["un_un", "tag_un", "un_cat", "tag_cat"]:
            row += f" & {{{model.lower()}_random_{dataset}_{setting}}}"
    latex_template += row + "\n"

latex_template += "\\midrule\n"

for model in ["SASRec", "BERT4Rec", "GRU4Rec"]:
    row = f"{model} (educated)"
    for dataset in ["steam", "ml-1m", "ml-100k"]:
        for setting in ["un_un", "tag_un", "un_cat", "tag_cat"]:
            if setting in ["un_un", "un_cat"]:
                row += f" & N.A."
            else:
                row += f" & {{{model.lower()}_educated_{dataset}_{setting}}}"
    latex_template += row + "\n"

latex_template += """\\bottomrule
\\end{tabular}
\\caption{Hamming Distance Metric: We report the Hamming distance values for all cases, where smaller values indicate better performance. un\\_un, tag\\_un, un\\_cat, tag\\_cat indicates respectively Untargeted-Uncategorized, Targeted-Categorized, Untargeted-Categorized and Targeted-Categorized respectively. N.A. denotes instances where the baseline method do not apply in that scenario.}
\\label{tab:edit_distance}
\\end{table*}
"""

# Map generation strategies to LaTeX categories
generation_strategy_mapping = {
    "genetic": "un_un",
    "genetic_categorized": "un_cat",
    "targeted": "tag_cat",
    "targeted_uncategorized": "tag_un",
}


# Replace placeholders in LaTeX template with num_edits
def replace_placeholder(latex_template, df):
    for _, row in df.iterrows():
        gen_str = row["generation_strategy"]
        model = row["model"]
        dataset = row["dataset"]
        num_edits = row["num_edits"] if "num_edits" in df.columns else row["edit_distance"]
        baseline = row["baseline"] if "baseline" in df.columns else "gece"

        # Get LaTeX category
        category = generation_strategy_mapping.get(gen_str)

        # Identify the corresponding cell and replace it
        latex_template = latex_template.replace(
            f"{{{model.lower()}_{baseline}_{dataset}_{category}}}",
            str(round(num_edits, 3)),
        )

    return latex_template


latex_filled = replace_placeholder(latex_template, df_fidel)
latex_filled = replace_placeholder(latex_filled, df_base)

print(latex_filled)
