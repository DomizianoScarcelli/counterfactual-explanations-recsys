import pandas as pd
from tempfile import NamedTemporaryFile
from performance_evaluation.alignment.utils import log_run, pk_exists

from config import GENERATIONS, HALLOFFAME_RATIO, POP_SIZE, DETERMINISM, MODEL, DATASET, ALLOWED_MUTATIONS, TIMESTAMP

configs = {
        "determinism": [DETERMINISM],
        "model": [MODEL],
        "dataset": [DATASET],
        "generations": [GENERATIONS],
        "halloffame_ratio": [HALLOFFAME_RATIO],
        "allowed_mutations": [tuple(ALLOWED_MUTATIONS)],
        "timestamp": [TIMESTAMP]}

class TestLogRun:
    def test_LogRun_DoesNotAddRecord_WhenPrimaryKeyAlreadyExists(self):
        prev_data = {
            "id": [1, 2],
            "value": ["existing_value1", "existing_value2"],
        }
        prev_data_config = {**prev_data}
        # Replicate each config value across all rows of the DataFrame
        for key, value in configs.items():
            prev_data_config[key] = [value[0]] * len(prev_data_config["id"])
        prev_df_no_config = pd.DataFrame(prev_data)
        prev_df_config = pd.DataFrame(prev_data_config)

        log = {
            "id": 1,
            "value": "existing_value1", 
        }

        primary_key = ["id"] 
        save_path = NamedTemporaryFile(delete=False).name  

        updated_df_config = log_run(prev_df_config, log, save_path, primary_key.copy(), add_config=True)
        updated_df_no_config = log_run(prev_df_no_config, log, save_path, primary_key, add_config=False)

        assert len(updated_df_config) == len(prev_df_config), "The record count should not increase if the primary key exists."
        assert len(updated_df_no_config) == len(prev_df_no_config), "The record count should not increase if the primary key exists."

        pd.testing.assert_frame_equal(updated_df_config, prev_df_config, check_dtype=False)
        pd.testing.assert_frame_equal(updated_df_no_config, prev_df_no_config, check_dtype=False)

        print("Test passed: log_run does not add a record when the primary key already exists.")

    def test_LogRun_AddRecord_WhenPrimaryKeyDoesNotExist(self):
        # Mock previous DataFrame
        prev_data = {
            "id": [1, 2],
            "value": ["existing_value1", "existing_value2"],
        }
        prev_data_config = {**prev_data}
        for key, value in configs.items():
            prev_data_config[key] = [value[0]] * len(prev_data_config["id"])
        prev_df_no_config = pd.DataFrame(prev_data)
        prev_df_config = pd.DataFrame(prev_data_config)

        # Log with a new record that does not match any existing primary key
        log = {
            "id": 3,  # New primary key
            "value": "new_value3"
        }
        primary_key = ["id"]

        save_path = NamedTemporaryFile(delete=False).name  

        # Run the log_run function
        updated_df_config = log_run(prev_df_config, log, save_path, primary_key.copy(), add_config=True)
        updated_df_no_config = log_run(prev_df_no_config, log, save_path, primary_key, add_config=False)

        # Assert that the record count has increased by 1
        assert len(updated_df_config) == len(prev_df_config) + 1, "The record count should increase when a new primary key is added."
        assert len(updated_df_no_config) == len(prev_df_no_config) + 1, "The record count should increase when a new primary key is added."

        # Assert that the new record exists in the updated DataFrame
        expected_data_config = {
            "id": [1, 2, 3],
            "value": ["existing_value1", "existing_value2", "new_value3"],
        }
        for key, value in configs.items():
            expected_data_config[key] = [value[0]] * len(expected_data_config["id"])
        expected_df_config = pd.DataFrame(expected_data_config)

        expected_data_no_config = {
            "id": [1, 2, 3],
            "value": ["existing_value1", "existing_value2", "new_value3"],
        }
        expected_df_no_config = pd.DataFrame(expected_data_no_config)

        pd.testing.assert_frame_equal(updated_df_config.reset_index(drop=True), expected_df_config, check_dtype=False)
        pd.testing.assert_frame_equal(updated_df_no_config.reset_index(drop=True), expected_df_no_config, check_dtype=False)

        print("Test passed: log_run adds a record when the primary key does not exist.")

    def test_LogRun_DoesNotAddRecord_WhenCompositePrimaryKeyAlreadyExists(self):
        prev_data = {
            "id": [1, 1, 2],
            "sub_id": [10, 20, 30],
            "value": ["existing_value1", "existing_value2", "existing_value3"],
        }
        prev_data_config = {**prev_data}
        for key, value in configs.items():
            prev_data_config[key] = [value[0]] * len(prev_data_config["id"])
        prev_df_no_config = pd.DataFrame(prev_data)
        prev_df_config = pd.DataFrame(prev_data_config)

        log = {
            "id": 1,
            "sub_id": 10,
            "value": "existing_value1",
        }

        primary_key = ["id", "sub_id"]

        save_path = NamedTemporaryFile(delete=False).name  

        updated_df_config = log_run(prev_df_config, log, save_path, primary_key.copy(), add_config=True)
        updated_df_no_config = log_run(prev_df_no_config, log, save_path, primary_key, add_config=False)

        assert len(updated_df_config) == len(prev_df_config), "The record count should not increase if the composite primary key exists."
        assert len(updated_df_no_config) == len(prev_df_no_config), "The record count should not increase if the composite primary key exists."

        pd.testing.assert_frame_equal(updated_df_config, prev_df_config, check_dtype=False)
        pd.testing.assert_frame_equal(updated_df_no_config, prev_df_no_config, check_dtype=False)

        print("Test passed: log_run does not add a record when the composite primary key already exists.")

    def test_LogRun_AddRecord_WhenCompositePrimaryKeyDoesNotExist(self):
        prev_data = {
            "id": [1, 1, 2],
            "sub_id": [10, 20, 30],
            "value": ["existing_value1", "existing_value2", "existing_value3"],
        }
        prev_data_config = {**prev_data}
        for key, value in configs.items():
            prev_data_config[key] = [value[0]] * len(prev_data_config["id"])
        prev_df_no_config = pd.DataFrame(prev_data)
        prev_df_config = pd.DataFrame(prev_data_config)

        log = {
            "id": 1,
            "sub_id": 40,
            "value": "new_value4",
        }

        primary_key = ["id", "sub_id"]

        save_path = NamedTemporaryFile(delete=False).name  

        updated_df_config = log_run(prev_df_config, log, save_path, primary_key.copy(), add_config=True)
        updated_df_no_config = log_run(prev_df_no_config, log, save_path, primary_key, add_config=False)

        assert len(updated_df_config) == len(prev_df_config) + 1, "The record count should increase when a new composite primary key is added."
        assert len(updated_df_no_config) == len(prev_df_no_config) + 1, "The record count should increase when a new composite primary key is added."

        expected_data_config = {
            "id": [1, 1, 2, 1],
            "sub_id": [10, 20, 30, 40],
            "value": ["existing_value1", "existing_value2", "existing_value3", "new_value4"],
        }
        for key, value in configs.items():
            expected_data_config[key] = [value[0]] * len(expected_data_config["id"])
        expected_df_config = pd.DataFrame(expected_data_config)

        expected_data_no_config = {
            "id": [1, 1, 2, 1],
            "sub_id": [10, 20, 30, 40],
            "value": ["existing_value1", "existing_value2", "existing_value3", "new_value4"],
        }
        expected_df_no_config = pd.DataFrame(expected_data_no_config)

        pd.testing.assert_frame_equal(updated_df_config.reset_index(drop=True), expected_df_config, check_dtype=False)
        pd.testing.assert_frame_equal(updated_df_no_config.reset_index(drop=True), expected_df_no_config, check_dtype=False)

        print("Test passed: log_run adds a record when the composite primary key does not exist.")


class TestPkExists:
    def test_PkExists_ReturnsTrue_WhenKeyExists(self):
        prev_data = {
            "id": [1, 1],
            "value": ["existing_value1", "existing_value2"],
        }
        prev_data_with_config = {**prev_data}
        for key, value in configs.items():
            prev_data_with_config[key] = [value[0]] * len(prev_data["id"])

        prev_df = pd.DataFrame(prev_data)
        prev_df_with_config = pd.DataFrame(prev_data_with_config)
        primary_key = ["id"]

        assert pk_exists(prev_df, primary_key, consider_config=False), "pk_exists should return True without considering config when the key exists."
        assert pk_exists(prev_df_with_config, primary_key, consider_config=True), "pk_exists should return True considering config when the key exists."
        print("Test passed: pk_exists returns True when the key exists, for both consider_config=True and False.")


    def test_PkExists_ReturnsFalse_WhenKeyDoesNotExist(self):
        prev_data = {
            "id": [1, 2],
            "value": ["existing_value1", "existing_value2"],
        }
        prev_data_with_config = {**prev_data}
        for key, value in configs.items():
            prev_data_with_config[key] = [value[0]] * len(prev_data["id"])

        prev_df = pd.DataFrame(prev_data)
        prev_df_with_config = pd.DataFrame(prev_data_with_config)
        primary_key = ["id"]

        assert not pk_exists(prev_df, primary_key, consider_config=False), "pk_exists should return False without considering config when the key does not exist."
        assert not pk_exists(prev_df_with_config, primary_key, consider_config=True), "pk_exists should return False considering config when the key does not exist."
        print("Test passed: pk_exists returns False when the key does not exist, for both consider_config=True and False.")


    def test_PkExists_ReturnsTrue_WhenCompositeKeyExists(self):
        prev_data = {
            "id": [1, 1, 2],
            "sub_id": [10, 10, 30],
            "value": ["existing_value1", "existing_value2", "existing_value3"],
        }
        prev_data_with_config = {**prev_data}
        for key, value in configs.items():
            prev_data_with_config[key] = [value[0]] * len(prev_data["id"])

        prev_df = pd.DataFrame(prev_data)
        prev_df_with_config = pd.DataFrame(prev_data_with_config)
        primary_key = ["id", "sub_id"]

        assert pk_exists(prev_df, primary_key, consider_config=False), "pk_exists should return True without considering config when the composite key exists."
        assert pk_exists(prev_df_with_config, primary_key, consider_config=True), "pk_exists should return True considering config when the composite key exists."
        print("Test passed: pk_exists returns True when the composite key exists, for both consider_config=True and False.")


    def test_PkExists_ReturnsFalse_WhenCompositeKeyDoesNotExist(self):
        prev_data = {
            "id": [1, 1, 2],
            "sub_id": [10, 20, 30],
            "value": ["existing_value1", "existing_value2", "existing_value3"],
        }
        prev_data_with_config = {**prev_data}
        for key, value in configs.items():
            prev_data_with_config[key] = [value[0]] * len(prev_data["id"])

        prev_df = pd.DataFrame(prev_data)
        prev_df_with_config = pd.DataFrame(prev_data_with_config)
        primary_key = ["id", "sub_id"]

        assert not pk_exists(prev_df, primary_key.copy(), consider_config=False), "pk_exists should return False without considering config when the composite key does not exist."
        assert not pk_exists(prev_df_with_config, primary_key, consider_config=True), "pk_exists should return False considering config when the composite key does not exist."
        print("Test passed: pk_exists returns False when the composite key does not exist, for both consider_config=True and False.")

