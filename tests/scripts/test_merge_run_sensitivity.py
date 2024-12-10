import pandas as pd
from scripts.merge_run_sensitivity import main as main_merge


def test_MergeRunSensitivity_ReturnsCorrectResult_WhenMetricsAreAllEqual():
    df_run = pd.DataFrame(
        {
            "i": [0, 0, 1, 1],
            "split": [
                "(None, 10, 0)",
                "(None, 10, 5)",
                "(None, 10, 0)",
                "(None, 5, 0)",
            ],
            "source": ["1,2,3", "1,2,3", "1,2,4", "1,2,4"],
        }
    )

    df_stats = pd.DataFrame(
        {
            "i": [0] * 4 + [1] * 4,
            "position": [10, 3, 4, 7] * 2,
            "metric1": [3.5] * 8,
            "metric2": [1.2] * 8,
            "source": ["1,2,3"] * 4 + ["1,2,4"] * 4,
        }
    )

    expected = pd.DataFrame(
        {
            "i": [0, 0, 1, 1],
            "split": [
                "(None, 10, 0)",
                "(None, 10, 5)",
                "(None, 10, 0)",
                "(None, 5, 0)",
            ],
            "source": ["1,2,3", "1,2,3", "1,2,4", "1,2,4"],
            "metric1": [3.5] * 4,
            "metric2": [1.2] * 4,
        }
    )

    result = main_merge(
        df_run=df_run,
        df_stats=df_stats,
        metrics=["metric1", "metric2"],
        on=("source", "source"),
        save_path=None,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_MergeRunSensitivity_ReturnsCorrectResult_WhenMetricsAreDifferent():
    df_run = pd.DataFrame(
        {
            "i": [0, 0, 1, 1],
            "split": [
                "(None, 10, 0)",
                "(None, 10, 5)",
                "(None, 10, 0)",
                "(None, 5, 0)",
            ],
            "source": ["1,2,3", "1,2,3", "1,2,4", "1,2,4"],
        }
    )

    df_stats = pd.DataFrame(
        {
            "i": [0] * 4 + [1] * 4,
            "position": [10, 3, 4, 7] * 2,
            "metric1": [3.5, 2.1, 4.0, 1.8, 3.0, 2.5, 4.2, 1.6],
            "metric2": [1.2, 0.8, 1.5, 1.1, 0.9, 0.7, 1.4, 1.0],
            "source": ["1,2,3"] * 4 + ["1,2,4"] * 4,
        }
    )

    print("Df run:", df_run)
    print("Df stats:", df_stats)

    expected = pd.DataFrame(
        {
            "i": [0, 0, 1, 1],
            "split": [
                "(None, 10, 0)",
                "(None, 10, 5)",
                "(None, 10, 0)",
                "(None, 5, 0)",
            ],
            "source": ["1,2,3", "1,2,3", "1,2,4", "1,2,4"],
            "metric1": [2.85, 2.65, 2.825, 3.35],
            "metric2": [1.15, 1.15, 1.0, 1.05]
        }
    )

    result = main_merge(
        df_run=df_run,
        df_stats=df_stats,
        metrics=["metric1", "metric2"],
        on=("source", "source"),
        save_path=None,
    )

    # Use assert_frame_equal to compare DataFrames
    pd.testing.assert_frame_equal(result, expected)
