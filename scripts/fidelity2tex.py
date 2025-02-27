from typing import Optional
import re

import fire
import pandas as pd

from constants import cat2id
from utils import load_log

strategy_mapping = {
    "targeted": "Targeted-Categorized",
    "targeted_uncategorized": "Targeted-Uncategorized",
    "genetic": "Untargeted-Uncategorized",
    "genetic_categorized": "Untargeted-Categorized",
}


def csv2table(log_path):
    # Read the CSV file
    df = load_log(log_path)

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
        "avg_gen_cost_at_1": "GENE distance_at_1",
        "avg_gen_cost_at_5": "GENE distance_at_5",
        "avg_gen_cost_at_10": "GENE distance_at_10",
        "avg_gen_cost_at_20": "GENE distance_at_20",
        "avg_cost_at_1": "PACE distance_at_1",
        "avg_cost_at_5": "PACE distance_at_5",
        "avg_cost_at_10": "PACE distance_at_10",
        "avg_cost_at_20": "PACE distance_at_20",
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
            "PACE distance_at_1",
            "PACE distance_at_5",
            "PACE distance_at_10",
            "PACE distance_at_20",
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
            "GENE distance_at_1",
            "GENE distance_at_5",
            "GENE distance_at_10",
            "GENE distance_at_20",
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
    distance_columns = [
        "distance_at_1",
        "distance_at_5",
        "distance_at_10",
        "distance_at_20",
    ]
    table = [*fidelity_columns, *distance_columns]
    df[table] = df[table].round(3)

    # Reorder columns according to the specified order
    column_order = [
        "Target",
        "Model",
        "Dataset",
        "Method",
        *table,
        "Generation Strategy",
    ]
    df = df[column_order]

    # Sort by Target, Model, Dataset, and alternate GENE/PACE methods
    df = df.sort_values(
        by=["Dataset", "Target", "Model", "Method"],
        key=lambda x: pd.Categorical(x, ["GENE", "PACE"]) if x.name == "Method" else x,
    )

    for strategy in strategy_mapping:
        strategy_group = df[df["Generation Strategy"] == strategy]
        df2latex(
            strategy_group,
            f"script_results/fidelity_results_{strategy}.tex",
            drop_last_col=True,
            strategy=strategy,
        )

    # Compute Targeted Average Tables
    targeted_avg = (
        df.groupby(["Model", "Dataset", "Method", "Generation Strategy"])[table]
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
    targeted_uncategorized_avg = targeted_uncategorized_avg.drop(
        "Generation Strategy", axis=1
    )
    targeted_categorized_avg = targeted_categorized_avg.drop(
        "Generation Strategy", axis=1
    )
    df2latex(
        targeted_categorized_avg,
        f"script_results/avg_fidelity_results_targeted_cat.tex",
    )
    df2latex(
        targeted_uncategorized_avg,
        f"script_results/avg_fidelity_results_targeted_uncat.tex",
    )


def df2latex(
    df: pd.DataFrame,
    output_file: str,
    drop_last_col: bool = False,
    strategy: Optional[str] = None,
):
    """
    Convert a Markdown table to a LaTeX longtable and save it to a file.

    :param md_table: A string containing the Markdown table.
    :param output_file: The file path where the LaTeX table should be saved.
    """
    if drop_last_col:
        df = df.iloc[:, :-1]  # Remove last column (strategy)
    df.columns = df.columns.str.strip()  # Clean column names

    # Begin LaTeX table
    latex_table = (
        r"""
    \begingroup
    \setlength{\tabcolsep}{4pt} % Adjust column spacing
    \renewcommand{\arraystretch}{1.5} % Adjust row height
    \small % Reduce font size
    \begin{longtable}{|l|l|l|l|l|l|l|l|l|l|l|l|}
    \hline
    \textbf{Target} & \textbf{Model} & \textbf{Dataset} & \textbf{Method} & \multicolumn{4}{c|}{\textbf{Fidelity@k}} & \multicolumn{4}{c|}{\textbf{Distance@k}} \\ \cline{5-12}
    & & & & @1 & @5 & @10 & @20 & @1 & @5 & @10 & @20 \\ \hline""".strip()
        + "\n"
    )

    # Add data rows
    i = 0
    for _, row in df.iterrows():
        row_values = " & ".join(map(str, row.values)) + " \\\\"
        if row["Method"] == "PACE":
            row_values += "\hline"

        if i > 0 and i < df.shape[0] - 1:
            prev_row = df.iloc[i+1]
            if "Target" in row and prev_row["Target"] != row["Target"]:
                row_values += "\hline"
            if prev_row["Dataset"] != row["Dataset"]:
                row_values += r"\multicolumn{{12}}{{|c|}}{{\textbf{{Results for {dataset} Dataset}}}} \\\hline".format(dataset=prev_row["Dataset"])

        row_values += "\n"
        latex_table += row_values
        i += 1

    # Add caption and closing
    footer= r"""
    \caption{{Model fidelity in the {caption}}}
    \label{{tab:{tab_name}}}
    \end{{longtable}}
    \endgroup""".format(
        caption=strategy_mapping[strategy] if strategy else "TODO",
        tab_name=strategy if strategy else "TODO",
    ).strip()
    latex_table += footer

    # Save to file
    with open(output_file, "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {output_file}")


def main(log_path: str):
    csv2table(log_path)


if __name__ == "__main__":
    fire.Fire(main)
