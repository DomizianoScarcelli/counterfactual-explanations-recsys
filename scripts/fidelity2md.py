import re

import fire
import pandas as pd

from constants import cat2id
from utils import load_log


def csv2table(log_path, output: str):
    # Read the CSV file
    df = load_log(log_path)

    print(df.columns)

    # Extract relevant columns
    columns_to_keep = {
        "model": "Model",
        "dataset": "Dataset",
        "generation_strategy": "Generation Strategy",
        "gen_target_y_at_1": "Target",
        "fidelity_score_at_1": "PACE fidelity_at_1",
        "fidelity_score_at_5": "PACE fidelity_at_5",
        "fidelity_score_at_10": "PACE fidelity_at_10",
        "fidelity_score_at_20": "PACE fidelity_at_20",
        "fidelity_gen_score_at_1": "GENE fidelity_at_1",
        "fidelity_gen_score_at_5": "GENE fidelity_at_5",
        "fidelity_gen_score_at_10": "GENE fidelity_at_10",
        "fidelity_gen_score_at_20": "GENE fidelity_at_20",
        "gen_cost": "GENE distance",
        "cost": "PACE distance",
        "gen_dataset_time": "Dataset Generation Time",
        "align_time": "Constraint A* Time",
        "count": "#users",
    }

    df = df[list(columns_to_keep.keys())].rename(columns=columns_to_keep)

    # Convert target
    id2cat = {v: k for k, v in cat2id.items()}

    def convert_target(target):
        match = re.match(r"^\{(\d+)\}$", str(target))
        if match:
            category_id = int(match.group(1))
            return id2cat.get(category_id, target)
        return target

    df["Target"] = df["Target"].apply(convert_target)

    # Create separate dataframes for PACE and GENE metrics
    pace_df = df[
        [
            "Model",
            "Dataset",
            "Target",
            "Generation Strategy",
            # "#users",
            "PACE fidelity_at_1",
            "PACE fidelity_at_5",
            "PACE fidelity_at_10",
            "PACE fidelity_at_20",
            "PACE distance",
            "Dataset Generation Time",
            "Constraint A* Time",
        ]
    ].copy()
    pace_df["Method"] = "PACE"
    pace_df.columns = [col.replace("PACE ", "") for col in pace_df.columns]

    gene_df = df[
        [
            "Model",
            "Dataset",
            "Target",
            "Generation Strategy",
            # "#users",
            "GENE fidelity_at_1",
            "GENE fidelity_at_5",
            "GENE fidelity_at_10",
            "GENE fidelity_at_20",
            "GENE distance",
            "Dataset Generation Time",
            "Constraint A* Time",
        ]
    ].copy()
    gene_df["Method"] = "GENE"
    gene_df.columns = [col.replace("GENE ", "") for col in gene_df.columns]

    # Combine the dataframes
    df = pd.concat([pace_df, gene_df], ignore_index=True)

    # Round fidelity scores.
    fidelity_columns = [
        "fidelity_at_1",
        "fidelity_at_5",
        "fidelity_at_10",
        "fidelity_at_20",
    ]
    df[fidelity_columns] = df[fidelity_columns].round(3)

    # Reorder columns according to the specified order
    column_order = [
        "Target",
        "Model",
        "Dataset",
        "Method",
        "fidelity_at_1",
        "fidelity_at_5",
        "fidelity_at_10",
        "fidelity_at_20",
        # "#users",
        "Generation Strategy",
    ]
    df = df[column_order]

    # Sort by Target, Model, Dataset, and alternate GENE/PACE methods
    df = df.sort_values(
        by=["Target", "Model", "Dataset", "Method"],
        key=lambda x: pd.Categorical(x, ["GENE", "PACE"]) if x.name == "Method" else x,
    )

    print(f"[DEBUG] strategies are", df["Generation Strategy"].unique())

    # Function to blank out repeated values in specified columns
    def blank_repeated_values(df_group):
        result_df = df_group.copy()
        non_metric_columns = ["Target", "Model", "Dataset", "Generation Strategy"]
        for col in non_metric_columns:
            mask = result_df[col].duplicated(keep="first")
            result_df.loc[mask, col] = ""
        return result_df

    # Apply the blank_repeated_values function to each group
    # FIX: this line is broken, it removes some generation strategies
    # df = df.groupby(['Target', 'Model', 'Dataset', "Generation Strategy"]).apply(blank_repeated_values).reset_index(drop=True)

    print(f"[DEBUG] df is", df)
    print(f"[DEBUG] strategies are", df["Generation Strategy"].unique())

    strategy_mapping = {
        "targeted": "Targeted-Categorized",
        "targeted_uncategorized": "Targeted-Uncategorized",
        "genetic": "Untargeted-Uncategorized",
        "genetic_categorized": "Untargeted-Categorized",
    }

    if output == "md":
        # Convert DataFrame to Markdown
        markdown_content = ""

        # Always create sections for all strategies, even if empty
        for strategy in strategy_mapping:
            markdown_content += f"### {strategy_mapping[strategy]}\n\n"
            strategy_group = df[df["Generation Strategy"] == strategy]
            if not strategy_group.empty:
                strategy_group = strategy_group.drop("Generation Strategy", axis=1)
                markdown_content += strategy_group.to_markdown(index=False) + "\n\n"
            else:
                markdown_content += "No data available for this strategy\n\n"

        with open("script_results/fidelity_results.md", "w") as f:
            f.write(markdown_content)
    elif output == "latex":
        for strategy in strategy_mapping:
            strategy_group = df[df["Generation Strategy"] == strategy]
            df2latex(strategy_group, f"script_results/fidelity_results_{strategy}.tex")

    # Compute Targeted Average Tables
    targeted_avg = (
        df.groupby(["Model", "Dataset", "Method", "Generation Strategy"])[
            fidelity_columns
        ]
        .mean()
        .reset_index()
    )
    targeted_categorized_avg = targeted_avg[
        targeted_avg["Generation Strategy"] == "targeted"
    ]
    targeted_uncategorized_avg = targeted_avg[
        targeted_avg["Generation Strategy"] == "targeted_uncategorized"
    ]
    targeted_categorized_avg = targeted_categorized_avg.round(3)
    targeted_uncategorized_avg = targeted_uncategorized_avg.round(3)
    targeted_uncategorized_avg  = targeted_uncategorized_avg.drop("Generation Strategy", axis=1)
    targeted_categorized_avg = targeted_categorized_avg.drop("Generation Strategy", axis=1)
    if output == "latex":
        df2latex(
            targeted_categorized_avg,
            f"script_results/avg_fidelity_results_targeted_cat.tex",
        )
        df2latex(
            targeted_uncategorized_avg,
            f"script_results/avg_fidelity_results_targeted_uncat.tex",
        )
    elif output == "md":
        markdown_content += f"### Average Targeted-Uncategorized"
        markdown_content += targeted_uncategorized_avg.to_markdown(index=False) + "\n\n"
        markdown_content += f"### Average Targeted-Categorized"
        markdown_content += targeted_categorized_avg.to_markdown(index=False) + "\n\n"
        with open("fidelity_results.md", "w") as f:
            f.write(markdown_content)


def df2latex(df: pd.DataFrame, output_file: str):
    """
    Convert a Markdown table to a LaTeX longtable and save it to a file.

    :param md_table: A string containing the Markdown table.
    :param output_file: The file path where the LaTeX table should be saved.
    """
    df = df.iloc[:, :-1]  # Remove last column (strategy)
    df.columns = df.columns.str.strip()  # Clean column names

    # Begin LaTeX table
    latex_table = (
        r"""
    \begingroup
    \setlength{\tabcolsep}{6pt} % Adjust column spacing
    \renewcommand{\arraystretch}{1.2} % Adjust row height
    \small % Reduce font size
    \begin{longtable}{|l|l|l|l|l|l|l|l|}
    \hline
    """.strip()
        + "\n"
    )

    # Add column headers
    headers = (
        " & ".join([f"\\textbf{{{col.replace('_at_', '@')}}}" for col in df.columns])
        + " \\\\ \hline\n"
    )
    latex_table += headers + "\\endhead\n"

    # Add data rows
    for _, row in df.iterrows():
        row_values = " & ".join(map(str, row.values)) + " \\\\"
        if row["Method"] == "PACE":
            row_values += "\hline"
        row_values += "\n"
        latex_table += row_values

    # Add caption and closing
    latex_table += r"""
    \caption{\textbf{TODO: these are still partial results, update!} Model fidelity in the Targeted-Categorized setting}
    \label{tab:eval_targ_cat}
    \end{longtable}
    \endgroup
    """.strip()

    # Save to file
    with open(output_file, "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {output_file}")


def main(log_path: str, output: str):
    csv2table(log_path, output)


if __name__ == "__main__":
    fire.Fire(main)
