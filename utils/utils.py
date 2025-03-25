import random
import time
from functools import wraps
from typing import Any, Callable, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor

from config.config import ConfigParams
from utils.RunLogger import RunLogger


def printd(statement, level=1):
    """
    Prints the statement only if the specified level is lower than the debug
    level.
    """
    if ConfigParams.DEBUG and level <= ConfigParams.DEBUG:
        print(statement)


class SeedSetter:
    # Class-level attribute to hold the single instance
    _instance = None
    previous_seed = None

    # Private constructor to ensure only one instance is created
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def set_seed(cls, seed: Optional[int] = None):
        if seed is None:
            seed = ConfigParams.SEED
        # Check if we are trying to change the seed once it's already set
        if cls.previous_seed is not None:
            raise ValueError(
                f"Seed has already been set to {cls.previous_seed}, cannot change it."
            )

        if not ConfigParams.DETERMINISM:
            return
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Store the seed as previous_seed
        cls.previous_seed = seed


def seq_tostr(seq: Tensor | List | Set) -> str:
    if isinstance(seq, Tensor):
        assert (
            seq.dim() == 1
        ), f"Sequence should be 1-dimensional, its shape is: {seq.shape}"
        seq = seq.tolist()

    return ",".join(str(x) for x in seq)


def timeit(func):
    """
    Decorator to measure and print the execution time of a function.

    Args:
        func (callable): The function to measure.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")
        return result

    return wrapper


def load_log(log_path) -> DataFrame:
    if log_path.endswith(".csv"):
        df = pd.read_csv(log_path)
    elif log_path.endswith(".db"):
        logger = RunLogger(log_path)
        df = logger.to_pandas(table="logs")
        logger.close()
    else:
        raise ValueError(f"Log must be .csv or .db, not .{log_path.split(".")[-1]}")
    return df


def infer_dtype(df: DataFrame) -> DataFrame:
    def infer(value):
        try:
            float_val = float(value)
            if float_val.is_integer():
                return int(float_val)
            return float_val
        except (ValueError, TypeError):
            if str(value).lower() in ["true", "false"]:  # Handle booleans
                return str(value).lower() == "true"
            return value

    for col in df.columns:
        df[col] = df[col].apply(infer)

    return df


class TimedFunction:
    """
    A wrapper class for timing the execution of a specific function.

    Attributes:
        func (Callable): The function to be timed.
        last_time (float): The time taken by the last execution of the function.

    Methods:
        __call__(*args, **kwargs): Executes the function, measuring the time taken.
        get_last_time(): Returns the time taken for the last function execution.
    """

    def __init__(self, func: Callable):
        """
        Initializes the TimedFunction with the function to be timed.

        Args:
            func (Callable): The function to wrap and time.
        """
        self.func = func
        self.last_time: float = 0.0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Calls the wrapped function and measures the time taken.

        Args:
            *args: Arguments to pass to the wrapped function.
            **kwargs: Keyword arguments to pass to the wrapped function.

        Returns:
            The result of the wrapped function.
        """
        start_time = time.time()
        result = self.func(*args, **kwargs)
        self.last_time = time.time() - start_time
        return result

    def get_last_time(self) -> float:
        """
        Returns the time taken by the last execution of the wrapped function.

        Returns:
            float: Time in seconds.
        """
        return self.last_time
