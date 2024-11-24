import pandas as pd
from tempfile import NamedTemporaryFile
from performance_evaluation.alignment.utils import log_run  # Replace with the actual module name

def test_LogRun_DoesNotAddRecord_WhenPrimaryKeyAlreadyExists():
    # Mock previous DataFrame
    prev_data = {
        "id": [1, 2],
        "value": ["existing_value1", "existing_value2"],
        "timestamp": ["2024-11-23T12:00:00", "2024-11-23T12:01:00"]
    }
    prev_df = pd.DataFrame(prev_data)

    # Log that matches an existing record based on primary_key
    log = {
        "id": 1,
        "value": "existing_value1", 
    }

    primary_key = ["id"] 
    # Temporary file to save the log
    save_path = NamedTemporaryFile(delete=False).name  

    # Run the log_run function
    updated_df = log_run(prev_df, log, save_path, primary_key, add_config=False)
    # Assert that the record count has not increased
    assert len(updated_df) == len(prev_df), "The record count should not increase if the primary key exists."

    # Assert that the DataFrame remains unchanged
    pd.testing.assert_frame_equal(updated_df, prev_df, check_dtype=False)

    print("Test passed: log_run does not add a record when the primary key already exists.")

def test_LogRun_AddRecord_WhenPrimaryKeyDoesNotExist():
    # Mock previous DataFrame
    prev_data = {
        "id": [1, 2],
        "value": ["existing_value1", "existing_value2"],
        "timestamp": ["2024-11-23T12:00:00", "2024-11-23T12:01:00"]
    }
    prev_df = pd.DataFrame(prev_data)

    # Log with a new record that does not match any existing primary key
    log = {
        "id": 3,  # New primary key
        "value": "new_value3"
    }
    primary_key = ["id"]

    # Temporary file to save the log
    save_path = NamedTemporaryFile(delete=False).name  

    # Run the log_run function
    updated_df = log_run(prev_df, log, save_path, primary_key, add_config=False)

    # Assert that the record count has increased by 1
    assert len(updated_df) == len(prev_df) + 1, "The record count should increase when a new primary key is added."

    # Assert that the new record exists in the updated DataFrame
    expected_data = {
        "id": [1, 2, 3],
        "value": ["existing_value1", "existing_value2", "new_value3"],
        "timestamp": ["2024-11-23T12:00:00", "2024-11-23T12:01:00", None]
    }
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(updated_df.reset_index(drop=True), expected_df, check_dtype=False)

    print("Test passed: log_run adds a record when the primary key does not exist.")


def test_LogRun_DoesNotAddRecord_WhenCompositePrimaryKeyAlreadyExists():
    # Mock previous DataFrame
    prev_data = {
        "id": [1, 1, 2],
        "sub_id": [10, 20, 30],
        "value": ["existing_value1", "existing_value2", "existing_value3"],
        "timestamp": ["2024-11-23T12:00:00", "2024-11-23T12:01:00", "2024-11-23T12:02:00"]
    }
    prev_df = pd.DataFrame(prev_data)

    # Log that matches an existing composite primary key
    log = {
        "id": 1,
        "sub_id": 10,
        "value": "existing_value1",
    }

    primary_key = ["id", "sub_id"]  # Composite primary key

    # Temporary file to save the log
    save_path = NamedTemporaryFile(delete=False).name  

    # Run the log_run function
    updated_df = log_run(prev_df, log, save_path, primary_key, add_config=False)
    
    # Assert that the record count has not increased
    assert len(updated_df) == len(prev_df), "The record count should not increase if the composite primary key exists."

    # Assert that the DataFrame remains unchanged
    pd.testing.assert_frame_equal(updated_df, prev_df, check_dtype=False)

    print("Test passed: log_run does not add a record when the composite primary key already exists.")

def test_LogRun_AddRecord_WhenCompositePrimaryKeyDoesNotExist():
    # Mock previous DataFrame
    prev_data = {
        "id": [1, 1, 2],
        "sub_id": [10, 20, 30],
        "value": ["existing_value1", "existing_value2", "existing_value3"],
        "timestamp": ["2024-11-23T12:00:00", "2024-11-23T12:01:00", "2024-11-23T12:02:00"]
    }
    prev_df = pd.DataFrame(prev_data)

    # Log with a new record that does not match any existing composite primary key
    log = {
        "id": 1,  # Partially overlapping key
        "sub_id": 40,  # New unique sub_id
        "value": "new_value4",
    }

    primary_key = ["id", "sub_id"]  # Composite primary key

    # Temporary file to save the log
    save_path = NamedTemporaryFile(delete=False).name  

    # Run the log_run function
    updated_df = log_run(prev_df, log, save_path, primary_key, add_config=False)

    # Assert that the record count has increased by 1
    assert len(updated_df) == len(prev_df) + 1, "The record count should increase when a new composite primary key is added."

    # Assert that the new record exists in the updated DataFrame
    expected_data = {
        "id": [1, 1, 2, 1],
        "sub_id": [10, 20, 30, 40],
        "value": ["existing_value1", "existing_value2", "existing_value3", "new_value4"],
        "timestamp": ["2024-11-23T12:00:00", "2024-11-23T12:01:00", "2024-11-23T12:02:00", None]
    }
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(updated_df.reset_index(drop=True), expected_df, check_dtype=False)

    print("Test passed: log_run adds a record when the composite primary key does not exist.")
