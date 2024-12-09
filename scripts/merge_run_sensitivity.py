import pandas as pd
import fire


def merge_run_sensitivity(evaluation_log_path: str, sens_stats_log_path: str):
    """
    Given an `evaluation_log_path` generated with `performance_evalutation.alignment.evaluate` and a `sens_stats_log_path` generated with `experiments.model_alignment`
    it joins the two logs (pandas Dataframes) into a single log, joining on the sequence.
    """
    # sens_df = pd.read_csv(sensitivity_log_path)
    eval_df = pd.read_csv(evaluation_log_path)
    stats_df = pd.read_csv(sens_stats_log_path)

    # TODO: take config in considerations, for now is ignored
    df = eval_df.merge(
        right=stats_df, how="left", left_on="source", right_on="sequence"
    )

    print(df)


if __name__ == "__main__":
    fire.Fire(merge_run_sensitivity)
