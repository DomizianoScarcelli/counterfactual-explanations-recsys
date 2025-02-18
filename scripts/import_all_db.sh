#!/bin/bash

# Set the target folder containing CSV files
CSV_FOLDER="results/evaluate/alignment"

# Set the database file (same for all CSVs)
DB_FILE="results/evaluate/alignment/alignment.db"


# Define the primary key as a properly formatted string
PRIMARY_KEY='["i", "gen_strategy", "split", "model", "dataset", "gen_target_y_at_1", "pop_size", "generations", "fitness_alpha", "include_sink", "mut_prob", "crossover_prob", "genetic_topk", "mutation_params", "ignore_genetic_split", "jaccard_threshold"]'

# Loop through each CSV file in the folder
for csv_file in "$CSV_FOLDER"/*.csv; do
    echo "Processing $csv_file ..."
    python -m scripts.dbcli import --csv-file="$csv_file" --db-file="$DB_FILE" --merge-cols=False --batch-size=512 --primary-key="$PRIMARY_KEY" --check-if-exists=False
done

echo "All CSV files have been processed and added to $DB_FILE."

