import os
from typing import List, Optional

import fire
import pandas as pd
from tqdm import tqdm

from config import ConfigParams
from utils_classes.RunLogger import RunLogger


def load_csv_db(
    csv_file: str,
    db_file: str,
    merge_cols: bool = True,
    primary_key: Optional[List[str]] = None,
    batch_size: int = 1,
):
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"The file {csv_file} does not exist.")

    # Read the CSV file using pandas
    df = pd.read_csv(csv_file)

    # Initialize the RunLogger to handle the database interactions
    logger = RunLogger(db_file, schema=None, add_config=False, merge_cols=merge_cols)

    # Initialize a list to collect rows for batch insert
    batch = []

    # Iterate through the rows and log them one by one
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Convert the row to a dictionary
        log_entry = row.to_dict()
        log_entry_metr = {
            k: v
            for k, v in log_entry.items()
            if k not in ConfigParams.configs_dict().keys()
        }
        log_entry_str = {
            k: str(v)
            for k, v in log_entry.items()
            if k in ConfigParams.configs_dict().keys()
        }
        log_entry = {**log_entry_metr, **log_entry_str}

        # Add the log entry to the batch
        batch.append(log_entry)

        # Once the batch reaches the specified size, insert all entries at once
        if len(batch) >= batch_size:
            # Insert the batch into the database
            for log in batch:
                if not logger.exists(
                    log=log,
                    primary_key=primary_key,
                    consider_config=False,
                    type_sensitive=False,
                ):
                    logger.log_run(log, primary_key=primary_key, strict=False)
            batch.clear()  # Reset the batch after inserting

    # Insert any remaining rows that did not fill the last batch
    if batch:
        for log in batch:
            if not logger.exists(
                log=log,
                primary_key=primary_key,
                consider_config=False,
                type_sensitive=False,
            ):
                logger.log_run(log, primary_key=primary_key, strict=False)

    print(
        f"CSV file '{csv_file}' has been successfully merged into SQLite DB '{db_file}'."
    )


if __name__ == "__main__":
    fire.Fire(load_csv_db)
