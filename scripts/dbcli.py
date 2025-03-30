import os
import sqlite3
from typing import List, Literal, Optional, TypeAlias

import fire
import pandas as pd
from tqdm import tqdm

from config.config import ConfigParams
from utils.utils import RunLogger

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
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"The file {csv_file} does not exist.")
        df = pd.read_csv(csv_file)
        logger = RunLogger(
            db_file, schema=None, add_config=False, merge_cols=merge_cols
        )
        batch = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
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
            batch.append(log_entry)
            if len(batch) >= batch_size:
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
        if not os.path.exists(db1):
            raise FileNotFoundError(f"{db1} doesn't exists")
        if not os.path.exists(db2):
            raise FileNotFoundError(f"{db2} doesn't exists")

        dir_path = os.path.dirname(db1)
        db1_name = os.path.splitext(os.path.basename(db1))[0]
        db2_name = os.path.splitext(os.path.basename(db2))[0]
        merged_db_name = f"{db1_name}_{db2_name}.db"
        merged_db_name = os.path.join(dir_path, merged_db_name)
        print(f"[DEBUG] merged db name:", merged_db_name)

        conn1 = sqlite3.connect(db1)
        print(f"Connected to db1 at {db1}")
        conn2 = sqlite3.connect(db2)
        print(f"Connected to db2 at {db2}")
        conn_merged = sqlite3.connect(merged_db_name)
        print(f"Connected to merged db at {merged_db_name}")

        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()

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
        confirm = input(f"Confirm CLEAN operation on db: {db}, type: {dbtype}? y/N: ")
        if confirm != "y":
            print("Aborted")
            return
        if dbtype == "alignment":
            primary_key = [
                "i",
                "gen_strategy",
                # "split",
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
                "jaccard_threshold",
            ]
            logger.dedupe(add_config=False, primary_key=primary_key)
            query = """DELETE FROM logs
            WHERE NOT (
                CAST(generations AS TEXT) = '10'
                AND CAST(halloffame_ratio AS TEXT) = '0'
                AND CAST(jaccard_threshold AS TEXT) = '0.5'
                AND CAST(fitness_alpha AS TEXT) = '0.5'
                AND CAST(mut_prob AS TEXT) = '0.5'
                AND CAST(pop_size AS TEXT) = '8192'
                AND CAST(generations AS TEXT) = '10'
                AND CAST(crossover_prob AS TEXT) = '0.7'
                AND CAST(genetic_topk AS TEXT) = '1'
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
            primary_key = [
                "source_sequence",
                "generation_strategy",
                "target_cat",
                "dataset",
                "model",
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
        else:
            raise ValueError(f"dbtype {dbtype} not supported")

    def recap(self, db: str, dbtype: DBType):
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.expand_frame_repr", False)
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
            query = """SELECT targeted, categorized, model, dataset, target, position, count(*) AS num_users FROM logs
            GROUP BY targeted, categorized, model, dataset, target, position
            ORDER BY targeted, categorized, model, dataset, target, CAST(position as INTEGER), num_users DESC;"""

            print(logger.query(query))
        elif dbtype == "automata_learning":
            query = """SELECT generation_strategy, model, dataset, target_cat, count(*) AS num_users FROM logs
            WHERE CAST(determinism AS TEXT) = 'True'
            AND CAST(generations AS TEXT) = '10'
            AND CAST(halloffame_ratio AS TEXT) = '0'
            AND CAST(fitness_alpha AS TEXT) = '0.5'
            AND CAST(mut_prob AS TEXT) = '0.5'
            AND CAST(pop_size AS TEXT) = '8192'
            AND CAST(generations AS TEXT) = '10'
            AND CAST(crossover_prob AS TEXT) = '0.7'
            AND CAST(genetic_topk AS TEXT) = '1'
            AND CAST(ignore_genetic_split AS TEXT) = 'True'
            GROUP BY generation_strategy, model, dataset, target_cat
            ORDER BY generation_strategy, model, dataset, target_cat, num_users DESC;"""

            print(logger.query(query))
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
