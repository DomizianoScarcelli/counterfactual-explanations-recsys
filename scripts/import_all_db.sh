#!/bin/bash

# Set the target folder containing CSV files
CSV_FOLDER=$1

# Set the database file (same for all CSVs)
DB_FILE=$2


# Loop through each CSV file in the folder
for csv_file in "$CSV_FOLDER"/*.csv; do
    echo "Processing $csv_file ..."
    python -m scripts.dbcli import_csv --csv-file="$csv_file" --db-file="$DB_FILE" --merge-cols=False --batch-size=512 --check-if-exists=False
done

echo "All CSV files have been processed and added to $DB_FILE."

