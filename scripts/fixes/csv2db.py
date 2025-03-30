from utils.utils import load_log
import fire
import pandas as pd
import sqlite3
import os


def main(db_path: str):
    df = load_log(db_path)
    save_to_sqlite(df, db_path)


def save_to_sqlite(df: pd.DataFrame, original_db_path: str):
    """Saves the DataFrame back to a new SQLite database."""
    db_name = os.path.splitext(os.path.basename(original_db_path))[0]
    new_db_path = f"{db_name}_uncorrupted.db"

    conn = sqlite3.connect(new_db_path)
    df.to_sql(
        "logs", conn, if_exists="replace", index=False
    )  # Adjust table name if needed
    conn.close()

    print(f"Database saved as: {new_db_path}")


if __name__ == "__main__":
    fire.Fire(main)
