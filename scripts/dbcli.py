import os
from typing import List, Optional, TypeAlias, Literal
import sqlite3

import fire
import pandas as pd
from tqdm import tqdm

from config import ConfigParams
from utils_classes.RunLogger import RunLogger

DBType: TypeAlias = Literal["sensitivity", "alignment", "automata_learning"]


class DBCLI:
    def import_csv(
        self,
        csv_file: str,
        db_file: str,
        merge_cols: bool = True,
        primary_key: Optional[List[str]] = None,
        batch_size: int = 10,
        check_if_exists: bool = True,
    ):
        # Check if the CSV file exists
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"The file {csv_file} does not exist.")

        # Read the CSV file using pandas
        df = pd.read_csv(csv_file)

        # Initialize the RunLogger to handle the database interactions
        logger = RunLogger(
            db_file, schema=None, add_config=False, merge_cols=merge_cols
        )

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
                    if check_if_exists and logger.exists(
                        log=log,
                        primary_key=primary_key,
                        consider_config=False,
                        type_sensitive=False,
                    ):
                        continue
                    logger.log_run(log, primary_key=primary_key, strict=False)
                batch.clear()  # Reset the batch after inserting

        # Insert any remaining rows that did not fill the last batch
        if batch:
            for log in batch:
                if check_if_exists and logger.exists(
                    log=log,
                    primary_key=primary_key,
                    consider_config=False,
                    type_sensitive=False,
                ):
                    continue
                logger.log_run(log, primary_key=primary_key, strict=False)

        print(
            f"CSV file '{csv_file}' has been successfully merged into SQLite DB '{db_file}'."
        )

    def merge(self, db1: str, db2: str):
        if not os.path.exists(db1) or not os.path.exists(db2):
            raise FileNotFoundError("One or both database files do not exist.")

        merged_db_name = f"{os.path.splitext(db1)[0]}_{os.path.splitext(db2)[0]}.db"
        conn1 = sqlite3.connect(db1)
        conn2 = sqlite3.connect(db2)
        conn_merged = sqlite3.connect(merged_db_name)

        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()
        # merged_cursor = conn_merged.cursor()

        cursor1.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables1 = {table[0] for table in cursor1.fetchall()}

        cursor2.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables2 = {table[0] for table in cursor2.fetchall()}

        common_tables = tables1.intersection(tables2)

        for table in common_tables:
            df1 = pd.read_sql_query(f"SELECT * FROM {table}", conn1)
            df2 = pd.read_sql_query(f"SELECT * FROM {table}", conn2)

            merged_df = pd.concat([df1, df2], ignore_index=True)
            merged_df.to_sql(table, conn_merged, if_exists="replace", index=False)

        conn1.close()
        conn2.close()
        conn_merged.close()

        print(f"Databases merged and saved as {merged_db_name}")

    def clean(self, db: str, dbtype: DBType):
        logger = RunLogger(db, schema=None, add_config=False)
        if dbtype == "alignment":
            primary_key = [
                "i",
                "gen_strategy",
                "split",
                "model",
                "dataset",
                "gen_target_y_at_1",
                "pop_size",
                "generations",
                "fitness_alpha",
                "include_sink",
                "mut_prob",
                "crossover_prob",
                "genetic_topk",
                "mutation_params",
                "ignore_genetic_split",
                "jaccard_threshold",
            ]
            logger.dedupe(add_config=False, primary_key=primary_key)
            query = """DELETE FROM logs
            WHERE NOT (
                CAST(split AS TEXT) = '(None, 10, 0)'
                AND CAST(determinism AS TEXT) = 'True'
                AND CAST(generations AS TEXT) = '10'
                AND CAST(halloffame_ratio AS TEXT) = '0'
                AND CAST(fitness_alpha AS TEXT) = '0.5'
                AND CAST(mut_prob AS TEXT) = '0.5'
                AND CAST(pop_size AS TEXT) = '8192'
                AND CAST(generations AS TEXT) = '10'
                AND CAST(crossover_prob AS TEXT) = '0.7'
                AND CAST(genetic_topk AS TEXT) = '1'
                AND CAST(ignore_genetic_split AS TEXT) = 'True'
            );"""

            logger.cursor.execute(query)
            logger.conn.commit()
        elif dbtype == "sensitivity":
            primary_key = [
                "i",
                "position",
                "targeted",
                "categorized",
                "target",
                "model",
                "dataset",
            ]
            logger.dedupe(add_config=False, primary_key=primary_key)
        elif dbtype == "automata_learning":
            raise NotImplementedError()
        else:
            raise ValueError(f"dbtype {dbtype} not supported")

    def recap(self, db: str, dbtype: DBType):
        logger = RunLogger(db, schema=None, add_config=False)
        if dbtype == "alignment":
            query = """SELECT generation_strategy, model, dataset, gen_target_y_at_1 AS target, count(*) AS num_users FROM logs
            WHERE CAST(split AS TEXT) = '(None, 10, 0)'
            AND CAST(determinism AS TEXT) = 'True'
            AND CAST(generations AS TEXT) = '10'
            AND CAST(halloffame_ratio AS TEXT) = '0'
            AND CAST(fitness_alpha AS TEXT) = '0.5'
            AND CAST(mut_prob AS TEXT) = '0.5'
            AND CAST(pop_size AS TEXT) = '8192'
            AND CAST(generations AS TEXT) = '10'
            AND CAST(crossover_prob AS TEXT) = '0.7'
            AND CAST(genetic_topk AS TEXT) = '1'
            AND CAST(ignore_genetic_split AS TEXT) = 'True'
            GROUP BY gen_target_y_at_1, model, dataset, generation_strategy
            ORDER BY generation_strategy, model, dataset, gen_target_y_at_1, num_users DESC;"""
            print(logger.query(query))
        elif dbtype == "sensitivity":
            raise NotImplementedError()
        elif dbtype == "automata_learning":
            raise NotImplementedError()
        else:
            raise ValueError(f"dbtype {dbtype} not supported")

    def check_dups(self, db: str, dbtype: DBType):
        logger = RunLogger(db, schema=None, add_config=False)
        if dbtype == "alignment":
            query = """WITH duplicate_counts AS (
            SELECT i, generation_strategy, model, dataset, gen_target_y_at_1, COUNT(*) AS dup_level
            FROM logs
            WHERE CAST(split AS TEXT) = '(None, 10, 0)'
              AND CAST(determinism AS TEXT) = 'True'
              AND CAST(generations AS TEXT) = '10'
              AND CAST(halloffame_ratio AS TEXT) = '0'
              AND CAST(fitness_alpha AS TEXT) = '0.5'
              AND CAST(mut_prob AS TEXT) = '0.5'
              AND CAST(pop_size AS TEXT) = '8192'
              AND CAST(generations AS TEXT) = '10'
              AND CAST(crossover_prob AS TEXT) = '0.7'
              AND CAST(genetic_topk AS TEXT) = '1'
              AND CAST(ignore_genetic_split AS TEXT) = 'True'
            GROUP BY i, gen_target_y_at_1, model, dataset, generation_strategy
            HAVING COUNT(*) > 1
        )
        SELECT dup_level, COUNT(*) AS num_dups
        FROM duplicate_counts
        GROUP BY dup_level
        ORDER BY dup_level;"""
            print(logger.query(query))
        elif dbtype == "sensitivity":
            raise NotImplementedError()
        elif dbtype == "automata_learning":
            raise NotImplementedError()
        else:
            raise ValueError(f"dbtype {dbtype} not supported")


if __name__ == "__main__":
    fire.Fire(DBCLI)
