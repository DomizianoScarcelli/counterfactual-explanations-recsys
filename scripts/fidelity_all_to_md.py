import re

import pandas as pd

from constants import cat2id


def csv_to_markdown(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extract relevant columns
    columns_to_keep = {
        "model": "Model",
        "dataset": "Dataset",
        "generation_strategy": "Generation Strategy",
        "gen_target_y@1": "Target",
        "fidelity_score@1": "PACE fidelity@1",
        "fidelity_score@5": "PACE fidelity@5",
        "fidelity_score@10": "PACE fidelity@10",
        "fidelity_score@20": "PACE fidelity@20",
        "fidelity_gen_score@1": "GENE fidelity@1",
        "fidelity_gen_score@5": "GENE fidelity@5",
        "fidelity_gen_score@10": "GENE fidelity@10",
        "fidelity_gen_score@20": "GENE fidelity@20",
        "gen_cost": "GENE distance",
        "cost": "PACE distance",
        "gen_dataset_time": "Dataset Generation Time",
        "align_time": "Constraint A* Time",
        "count": "#users"
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
    pace_df = df[[
        'Model', 'Dataset', 'Target', 'Generation Strategy', '#users',
        'PACE fidelity@1', 'PACE fidelity@5', 'PACE fidelity@10', 'PACE fidelity@20',
        'PACE distance', 'Dataset Generation Time', 'Constraint A* Time'
    ]].copy()
    pace_df['Method'] = 'PACE'
    pace_df.columns = [col.replace('PACE ', '') for col in pace_df.columns]


    gene_df = df[[
        'Model', 'Dataset', 'Target', 'Generation Strategy', '#users',
        'GENE fidelity@1', 'GENE fidelity@5', 'GENE fidelity@10', 'GENE fidelity@20',
        'GENE distance', 'Dataset Generation Time', 'Constraint A* Time'
    ]].copy()
    gene_df['Method'] = 'GENE'
    gene_df.columns = [col.replace('GENE ', '') for col in gene_df.columns]

    # Combine the dataframes
    df = pd.concat([pace_df, gene_df], ignore_index=True)

    # Convert fidelity scores to percentages
    fidelity_columns = ['fidelity@1', 'fidelity@5', 'fidelity@10', 'fidelity@20']
    df[fidelity_columns] = df[fidelity_columns].multiply(100).round(2)

    # Reorder columns according to the specified order
    column_order = [
        'Target', 'Model', 'Dataset', 'Method', 'fidelity@1', 'fidelity@5', 
        'fidelity@10', 'fidelity@20', '#users', 'Generation Strategy'
    ]
    df = df[column_order]

    # Sort by Target, Model, Dataset, and alternate GENE/PACE methods
    df = df.sort_values(
        by=['Target', 'Model', 'Dataset', 'Method'],
        key=lambda x: pd.Categorical(x, ['GENE', 'PACE']) if x.name == 'Method' else x
    )

    print(f"[DEBUG] strategies are", df["Generation Strategy"].unique())
    # Function to blank out repeated values in specified columns
    def blank_repeated_values(df_group):
        result_df = df_group.copy()
        non_metric_columns = ['Target', 'Model', 'Dataset', "Generation Strategy"]
        for col in non_metric_columns:
            mask = result_df[col].duplicated(keep='first')
            result_df.loc[mask, col] = ''
        return result_df

    # Apply the blank_repeated_values function to each group
    #FIX: this line is broken, it removes some generation strategies
    # df = df.groupby(['Target', 'Model', 'Dataset', "Generation Strategy"]).apply(blank_repeated_values).reset_index(drop=True)

    print(f"[DEBUG] df is", df)
    print(f"[DEBUG] strategies are", df["Generation Strategy"].unique())

    # Convert DataFrame to Markdown
    markdown_content = ""
    strategy_mapping = {
        "targeted": "Targeted-Categorized",
        "targeted_uncategorized": "Targeted-Uncategorized",
        "genetic": "Untargeted-Uncategorized",
        "genetic_categorized": "Untargeted-Categorized",
    }
    
    # Always create sections for all strategies, even if empty
    for strategy in strategy_mapping:
        markdown_content += f"### {strategy_mapping[strategy]}\n\n"
        strategy_group = df[df['Generation Strategy'] == strategy]
        if not strategy_group.empty:
            strategy_group = strategy_group.drop("Generation Strategy", axis=1)
            markdown_content += strategy_group.to_markdown(index=False) + "\n\n"
        else:
            #FIX: untargeted groups are empty, understand why
            markdown_content += "No data available for this strategy\n\n"

    with open("fidelity_results.md", "w") as f:
        f.write(markdown_content)


# Example usage
csv_file = "results/stats/alignment/fidelity_all.csv"
csv_to_markdown(csv_file)
