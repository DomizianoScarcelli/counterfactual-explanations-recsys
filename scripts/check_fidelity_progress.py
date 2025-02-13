import pandas as pd

def count_missing_rows(csv_filepath, config):
    # Load the CSV file
    df = pd.read_csv(csv_filepath)

    # Ensure 'gen_target_y@1' column exists, filling NaN with "N/A"
    if 'gen_target_y@1' in df.columns:
        df['gen_target_y@1'] = df['gen_target_y@1'].fillna('N/A')

    # Expected models
    models = ["BERT4Rec", "SASRec", "GRU4Rec"]

    # Convert DataFrame into a dictionary for easy lookups
    row_counts = {(row['model'], row['dataset'], row['generation_strategy'], row['gen_target_y@1']): row['count']
                  for _, row in df.iterrows()}

    # Compute missing rows
    missing_counts = {}
    for model in models:
        for dataset, strategy, expected_rows in config:
            # Find all rows that match (model, dataset, strategy)
            relevant_keys = [key for key in row_counts if key[:3] == (model, dataset, strategy)]

            if relevant_keys:
                for key in relevant_keys:
                    actual_count = row_counts[key]
                    missing = expected_rows - actual_count
                    missing_counts[key] = missing
            else:
                # No matching dataset+strategy+model found, assume all expected rows are missing
                missing_counts[(model, dataset, strategy, "N/A")] = expected_rows  

    return missing_counts

def pretty_print_results(missing_counts):
    totally_missing = []
    partially_missing = []
    ok = []
    more = []

    for (model, dataset, strategy, target), missing in missing_counts.items():
        target_str = f"{{{target}}}" if target != "N/A" else ""
        status = f"MISSING {missing}" if missing > 0 else "OK"
        if missing < 0:
            status = f"MORE {abs(missing)}"

        line = f'{model},{dataset},{strategy},{target_str},{missing}: {status}'

        if missing == 0:
            ok.append(line)
        elif missing > 0 and missing < config_dict[(dataset, strategy)]:
            partially_missing.append(line)
        elif missing > 0:
            totally_missing.append(line)
        else:  # missing < 0 case
            more.append(line)

    # Print results in sections
    print("----TOTALLY MISSING----")
    for line in totally_missing:
        print(line)

    print("\n----PARTIALLY MISSING----")
    for line in partially_missing:
        print(line)

    print("\n----OK----")
    for line in ok:
        print(line)

    print("\n----MORE----")
    for line in more:
        print(line)

# Example usage:
config = [
    ("ml-100k", "targeted", 943),
    ("ml-100k", "targeted_uncategorized", 943),
    ("ml-100k", "genetic", 943),
    ("ml-100k", "genetic_categorized", 943),
    ("ml-1m", "targeted", 200),
    ("ml-1m", "targeted_uncategorized", 200),
    ("ml-1m", "genetic", 200),
    ("ml-1m", "genetic_categorized", 200),
]
config_dict = {(dataset, strategy): expected for dataset, strategy, expected in config}

csv_filepath = "results/stats/alignment/fidelity_all.csv"
missing_rows = count_missing_rows(csv_filepath, config)
pretty_print_results(missing_rows)
