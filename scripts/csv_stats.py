"""
This is a CLI wrapper to invoke the `performance_evaluation.alignment.utils.get_log_stats` with custom parameters.
"""

import fire

from performance_evaluation.alignment.utils import get_log_stats

if __name__ == "__main__":
    fire.Fire(get_log_stats)




