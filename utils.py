import random
import time
from typing import Any, Callable

import numpy as np
import torch

from config import DEBUG, DETERMINISM


def printd(statement, level=1):
    """
    Prints the statement only if the specified level is lower than the debug 
    level.
    """
    if DEBUG and level <= DEBUG:
        print(statement)


def set_seed(seed: int = 42):
    # print(f"[DEBUG] Setting seed: {seed}")
    MAX_SEED = 2**32 - 1
    seed %= (MAX_SEED + 1)
    if not DETERMINISM:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
