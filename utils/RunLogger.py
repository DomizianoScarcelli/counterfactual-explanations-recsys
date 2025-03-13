import sqlite3
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Union
from dotenv import load_dotenv
import os

import pandas as pd
import libsql_experimental as libsql
from config.config import ConfigParams

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path)

TURSO_URL = os.getenv("TURSO_URL")
TURSO_TOKEN = os.getenv("TURSO_TOKEN")


class RunLogger:
    def __init__(
        self,
        db_path: Optional[Union[str, Path]],
        local: Optional[bool] = None,
        sync: Optional[bool] = False,
        schema: Optional[Dict[str, Any]] = None,
        add_config: bool = False,
        merge_cols: bool = True,  # Changed default to True since it's needed for schema-less operation
    ):
        self.db_path = db_path
        self.merge_cols = merge_cols
        self.schema = self._normalize_schema(schema) if schema else {}
        self.add_config = add_config
        self.local = local if local is not None else ConfigParams.LOCAL
        if not self.local:
            if sync:
                self.conn = libsql.connect(
                    "local.db", sync_url=TURSO_URL, auth_token=TURSO_TOKEN
                )
                self.conn.sync()
            else:
                self.conn = libsql.connect(TURSO_URL, auth_token=TURSO_TOKEN)
            self.cursor = self.conn.cursor()
        else:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
        self.blacklist = ["timestamp", "target_cat", "_id", "rowid"]

        # Add config parameters to schema if add_config is True
        if self.add_config and schema:  # Only add to schema if schema is provided
            configs = [
                key for key in ConfigParams.configs_dict().keys() if key != "timestamp"
            ]
            for config_key in configs:
                if config_key not in self.schema:
                    self.schema[config_key] = str

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

    def fetch_one(self, query, params=None):
        # if self.local:
        self.cursor.execute(query, params or ())
        return self.cursor.fetchone()
        # else:
        #     result = self.cursor.execute(query, params or []).rows
        #     return [result[0]] if result else None

    def _check_or_init_db(self):
        """Check if the database exists and initialize if needed."""
        table_exists = self.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='logs';"
        )

        if not table_exists:
            # If no schema provided, create an empty table with a temporary column
            # (SQLite requires at least one column when creating a table)
            if not self.schema:
                self.cursor.execute(
                    """
                    CREATE TABLE logs (
                        _id INTEGER PRIMARY KEY AUTOINCREMENT
                    )
                """
                )
            else:
                # Create table with provided schema
                self._init_table()

            self.conn.commit()
            return

        # If schema provided, check for missing columns
        if self.schema:
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
        """Initialize table with schema."""
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

    def log_run(
        self,
        log: Dict[str, Any],
        primary_key: Optional[List[str]] = None,
        strict: bool = False,
        tostr: bool = False,
    ):
        try:
            self._log_run(log, primary_key, strict, tostr)
        except sqlite3.OperationalError:
            print("DATABASE LOCKED; sleeping for 3 seconds and trying again")
            time.sleep(3)
            self.log_run(log, primary_key, strict, tostr)

    def _log_run(
        self,
        log: Dict[str, Any],
        primary_key: Optional[List[str]] = None,
        strict: bool = False,
        tostr: bool = False,
    ):
        # Normalize column names in the log
        log = {self._normalize_column_name(key): value for key, value in log.items()}

        if primary_key is None:
            primary_key = [key for key in log.keys() if key not in ["_id", "rowid"]]

        # Add config values to the log entry if add_config is True
        if self.add_config:
            configs = ConfigParams.configs_dict(pandas=False)
            for config_key, config_value in configs.items():
                log[self._normalize_column_name(config_key)] = str(config_value)
                if config_key not in self.blacklist:
                    primary_key.append(self._normalize_column_name(config_key))

        # Check for new columns and add them if merge_cols is True
        if self.merge_cols:
            self.cursor.execute("PRAGMA table_info(logs)")
            existing_columns = {row[1]: row[2] for row in self.cursor.fetchall()}
            new_columns = []

            for col_name, _ in log.items():
                if col_name not in existing_columns:
                    sql_type = self._get_sql_type(str)
                    new_columns.append((col_name, sql_type))

            # Add any new columns found
            for col_name, sql_type in new_columns:
                try:
                    self.cursor.execute(
                        f"ALTER TABLE logs ADD COLUMN {col_name} {sql_type};"
                    )
                    self.conn.commit()
                    print(f"Added new column during logging: {col_name} ({sql_type})")
                    self.schema[col_name] = type(log[col_name])
                except sqlite3.Error as e:
                    print(f"Error adding column {col_name}: {e}")

        # Get updated list of columns
        self.cursor.execute("PRAGMA table_info(logs)")
        existing_columns = {row[1] for row in self.cursor.fetchall()}
        primary_key = [k for k in primary_key if k in existing_columns]

        try:
            # Get count before insertion (strict mode check)
            count_before = None
            if strict:
                count_before = self.fetch_one("SELECT COUNT(*) FROM logs;")[0]

            if tostr:
                log = {k: str(v) for k, v in log.items()}
            columns = ", ".join(log.keys())
            placeholders = ", ".join(["?" for _ in log])
            values = tuple(log.values())

            self.cursor.execute(
                f"INSERT INTO logs ({columns}) VALUES ({placeholders})", values
            )
            self.conn.commit()

            # Get count after insertion (strict mode check)
            if strict:
                count_after = self.fetch_one("SELECT COUNT(*) FROM logs;")[0]

                if count_before is not None and count_after <= count_before:
                    print(
                        f"[ERROR] Row count mismatch! Expected {count_before + 1}, but got {count_after}."
                    )
                    print(f"Primary key: {primary_key}")
                    print(f"Log entry: {log}")
                    raise ValueError(
                        "Strict mode violation: Row count did not increase as expected."
                    )

        except sqlite3.Error as e:
            print(f"Error while executing SQL: {e}")
            print(f"Log entry: {log}")
            print(f"Primary key used: {primary_key}")
            raise

    def exists(
        self,
        log: Dict[str, Any],
        primary_key: Optional[List[str] | Set[str]],
        consider_config: bool = True,
        type_sensitive: bool = True,
        whitelist: List[str] = [],
    ) -> bool:
        """
        Checks if inserting the given log entry will violate the primary key constraint.

        Args:
            log: Dictionary containing the log entry
            primary_key: List of column names to use as primary key
            consider_config: If True, primary key is extended with config keys
            type_sensitive: If False, performs type-insensitive comparison (converts all values to strings)

        Returns:
            bool: True if a matching record exists, False otherwise
        """
        # Check if the table is empty
        if self.fetch_one("SELECT COUNT(*) FROM logs")[0] == 0:
            return False  # Table is empty, so no record can exist

        log = {self._normalize_column_name(k): v for k, v in log.items()}
        # blacklist = set(ConfigParams.configs_dict()) - set({"gen_target_y@1", "determinism", "generation_strategy", "allowed_mutations", "include_sink", "pop_size", "model"})
        if not primary_key:
            primary_key = [key for key in log if key not in self.blacklist]

        primary_key = set(primary_key)
        if consider_config:
            log.update(ConfigParams.configs_dict(pandas=False, tostr=True))
            primary_key |= set(ConfigParams.configs_dict())
        primary_key -= set(self.blacklist)
        primary_key |= set(whitelist)

        primary_key = list(primary_key)

        # Normalize column names for the primary key
        primary_key = [self._normalize_column_name(k) for k in primary_key]

        assert all(
            key in log for key in primary_key
        ), f"keys not in log: {primary_key}, {list(log.keys())}"
        if type_sensitive:
            # Original type-sensitive comparison
            key_values = tuple(log[k] for k in primary_key)
            key_str = " AND ".join([f"{k} = ?" for k in primary_key])
        else:
            # Type-insensitive comparison by converting everything to strings
            key_values = tuple(str(log[k]) for k in primary_key)
            key_str = " AND ".join([f"CAST({k} AS TEXT) = ?" for k in primary_key])

        return (
            self.fetch_one(f"SELECT 1 FROM logs WHERE {key_str}", key_values)
            is not None
        )

    def get_logs(self) -> pd.DataFrame:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.expand_frame_repr", False)
        return pd.read_sql("SELECT * FROM logs", self.conn)

    def query(self, query) -> pd.DataFrame:
        return pd.read_sql(query, self.conn)

    def dedupe(self, primary_key: Optional[List[str]] = None, add_config: bool = True):
        self.cursor.execute("PRAGMA table_info(logs)")
        existing_columns = {row[1] for row in self.cursor.fetchall()}

        if not primary_key:
            primary_key = [col for col in existing_columns if col not in self.blacklist]
        else:
            primary_key = [
                self._normalize_column_name(k)
                for k in primary_key
                if k in existing_columns
            ]

        if not primary_key:
            print("[ERROR] No valid primary key columns found for deduplication.")
            return

        if add_config:
            primary_key += [
                self._normalize_column_name(key)
                for key in ConfigParams.configs_dict()
                if key not in self.blacklist + primary_key
            ]

        print("Final primary key columns:", primary_key)

        # Ensure proper grouping and handle NULLs
        key_str = ", ".join([f"CAST({k} as TEXT)" for k in primary_key])

        total_before = self.fetch_one("SELECT COUNT(*) FROM logs")[0]

        self.cursor.execute(
            f"""
            DELETE FROM logs
            WHERE ROWID NOT IN (
                SELECT MIN(ROWID) 
                FROM logs
                GROUP BY {key_str}
            );
            """
        )

        self.conn.commit()  # Ensure changes are saved_models

        total_after = self.fetch_one("SELECT COUNT(*) FROM logs")[0]

        print(f"Removed {total_before - total_after} duplicates.")

    def to_pandas(self, table: str):
        from utils.utils import infer_dtype

        query = f"SELECT * FROM {table}"
        df = pd.read_sql_query(query, self.conn)
        df = infer_dtype(df)
        return df

    def close(self):
        self.conn.close()
