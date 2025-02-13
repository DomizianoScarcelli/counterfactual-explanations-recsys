import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from config import ConfigParams


class RunLogger:
    def __init__(
        self,
        db_path: Union[str, Path],
        schema: Dict[str, Any],
        add_config: bool = False,
        merge_cols: bool = False,
    ):
        self.db_path = db_path
        self.merge_cols = merge_cols
        self.schema = self._normalize_schema(
            schema
        )  # Normalize column names (replace @ with _at_)
        self.add_config = add_config
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Add config parameters to schema if add_config is True
        if self.add_config:
            configs = [
                key for key in ConfigParams.configs_dict().keys() if key != "timestamp"
            ]
            for config_key in configs:
                if config_key not in self.schema:
                    self.schema[config_key] = str  # Assuming all configs are strings

        self._check_or_init_db()

    def _normalize_column_name(self, column_name: str) -> str:
        """Normalize column names by replacing @ with _at_."""
        return column_name.replace("@", "_at_")

    def _normalize_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize schema to handle @ in column names."""
        return {
            self._normalize_column_name(key): value for key, value in schema.items()
        }

    def _get_sql_type(self, value: Any) -> str:
        if isinstance(value, int):
            return "INTEGER"
        if isinstance(value, bool):
            return "INTEGER"
        elif isinstance(value, float):
            return "REAL"
        elif isinstance(value, str):
            return "TEXT"
        else:
            return "TEXT"  # Default to TEXT for unknown types

    def _check_or_init_db(self):
        """Check if the database exists and matches the expected schema, or merge columns if enabled."""
        if Path(self.db_path).exists():
            # Check if the 'logs' table exists
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='logs';"
            )
            table_exists = self.cursor.fetchone()

            if table_exists:
                # If the table exists, fetch current schema
                self.cursor.execute("PRAGMA table_info(logs)")
                existing_columns = {row[1]: row[2] for row in self.cursor.fetchall()}
                expected_columns = {
                    key: self._get_sql_type(value) for key, value in self.schema.items()
                }

                # Determine missing columns
                missing_columns = [
                    key for key in expected_columns if key not in existing_columns
                ]

                if missing_columns and self.merge_cols:
                    self._add_missing_columns(missing_columns, expected_columns)
                elif missing_columns:
                    raise ValueError(
                        f"Database schema mismatch. Missing columns: {missing_columns}"
                    )

                return

        # If table does not exist, create it
        self._init_table()

    def _add_missing_columns(
        self, missing_columns: List[str], expected_columns: Dict[str, str]
    ):
        """Adds missing columns to the existing database schema."""
        for column in missing_columns:
            sql_type = expected_columns[column]
            self.cursor.execute(f"ALTER TABLE logs ADD COLUMN {column} {sql_type};")
        self.conn.commit()
        print(f"Added missing columns: {missing_columns}")

    def _init_table(self):
        columns = ", ".join(
            [f"{key} {self._get_sql_type(value)}" for key, value in self.schema.items()]
        )
        self.cursor.execute(
            f"""
            CREATE TABLE logs (
                {columns}
            )
        """
        )
        self.conn.commit()

    def log_run(self, log: Dict[str, Any], primary_key: Optional[List[str]] = None):
        # Normalize column names in the log
        log = {self._normalize_column_name(key): value for key, value in log.items()}

        if primary_key is None:
            primary_key = list(
                log.keys()
            )  # Use all columns as primary key if none is given

        # Add config values to the log entry if add_config is True
        if self.add_config:
            configs = ConfigParams.configs_dict()
            for config_key, config_value in configs.items():
                if config_key != "timestamp":  # Skip the timestamp key
                    log[self._normalize_column_name(config_key)] = str(
                        config_value[0]
                    )  # TODO: change this when the ConfigParams.configs_dict is changed from key:[value] to key:value

        self.cursor.execute("PRAGMA table_info(logs)")
        existing_columns = {row[1] for row in self.cursor.fetchall()}
        primary_key = [k for k in primary_key if k in existing_columns]

        try:
            if primary_key:
                key_values = tuple(log[k] for k in primary_key if k in log)
                key_str = " AND ".join([f"{k} = ?" for k in primary_key if k in log])

                self.cursor.execute(f"SELECT 1 FROM logs WHERE {key_str}", key_values)
                if not self.cursor.fetchone():
                    columns = ", ".join(log.keys())
                    placeholders = ", ".join(["?" for _ in log])
                    values = tuple(log.values())
                    self.cursor.execute(
                        f"INSERT INTO logs ({columns}) VALUES ({placeholders})", values
                    )
                    self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error while executing SQL: {e}")
            print(f"Log entry: {log}")
            print(f"Primary key used: {primary_key}")
            raise  # Re-raise the exception after logging the details

    def exists(self, key: str, value: Any, consider_config: bool = True) -> bool:
        """
        Check if a log entry exists in the database with the specified primary key.
        If consider_config is False, primary key is extended with config keys.
        """
        if not consider_config:
            primary_key = [key]
        else:
            primary_key = [key] + [
                config_key
                for config_key in ConfigParams.configs_dict().keys()
                if config_key != "timestamp"
            ]

        # Normalize column names for the primary key
        primary_key = [self._normalize_column_name(k) for k in primary_key]

        self.cursor.execute("PRAGMA table_info(logs)")
        existing_columns = {row[1] for row in self.cursor.fetchall()}
        primary_key = [k for k in primary_key if k in existing_columns]

        key_values = tuple(value for k in primary_key if k in existing_columns)
        key_str = " AND ".join(
            [f"{k} = ?" for k in primary_key if k in existing_columns]
        )

        self.cursor.execute(f"SELECT 1 FROM logs WHERE {key_str}", key_values)
        return self.cursor.fetchone() is not None

    def will_exist(
        self, log: Dict[str, Any], primary_key: List[str], consider_config: bool = True
    ) -> bool:
        """
        Checks if inserting the given log entry will violate the primary key constraint.
        If consider_config is False, primary key is extended with config keys.
        """
        if not consider_config:
            primary_key = [k for k in primary_key if k in log]
        else:
            primary_key = primary_key + [
                config_key
                for config_key in ConfigParams.configs_dict().keys()
                if config_key != "timestamp"
            ]

        # Normalize column names for the primary key
        primary_key = [self._normalize_column_name(k) for k in primary_key]

        key_values = tuple(log[k] for k in primary_key if k in log)
        key_str = " AND ".join([f"{k} = ?" for k in primary_key if k in log])

        self.cursor.execute(f"SELECT 1 FROM logs WHERE {key_str}", key_values)
        return self.cursor.fetchone() is not None

    def get_logs(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM logs", self.conn)

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    # Example Usage:
    run_log = {
        "i": int,
        "gen_strategy": str,
        "gen_error": float,
        "gen_source": str,
        "gen_aligned": str,
        "gen_alignment": str,
        "gen_cost": float,
        "gen_gt": str,
        "gen_aligned_gt": str,
        "gen_dataset_time": float,
        "gen_good_points_percentage": float,
        "gen_bad_points_percentage": float,
        "gen_good_points_edit_distance": float,
        "gen_bad_points_edit_distance": float,
    }

    logger = RunLogger("run_log.db", schema=run_log, add_config=True)
    log_entry = {"i": 1, "gen_strategy": "A", "gen_error": 0.2, "gen_source": "src"}
    log_entry_3 = {"i": 1, "gen_strategy": "A", "gen_error": 0.2, "gen_source": "src"}
    log_entry_2 = {"i": 2, "gen_strategy": "A", "gen_error": 0.2, "gen_source": "src"}
    logger.log_run(log_entry, primary_key=["i"])
    logger.log_run(log_entry_2, primary_key=["i"])
    logger.log_run(log_entry_3, primary_key=["i"])
    print(logger.get_logs())
