import random
import time
from typing import Any, Callable, List, Optional
from functools import wraps

import numpy as np
import torch
from torch import Tensor, Value

import zlib
from config import ConfigParams


def printd(statement, level=1):
    """
    Prints the statement only if the specified level is lower than the debug
    level.
    """
    if ConfigParams.DEBUG and level <= ConfigParams.DEBUG:
        print(statement)


def compute_seed(dependencies: List[list | tuple | int]) -> int:
    if any(x for x in dependencies if not isinstance(x, (list, tuple, int))):
        raise ValueError(f"Some dependencies are not list, tuple or ints")
    _bytes = [bytes(x) for x in dependencies if isinstance(x, (list, tuple))]
    count = _bytes[0]
    for b in _bytes[1:]:
        count += b
    other = sum(x for x in dependencies if isinstance(x, (int)))
    return zlib.crc32(count) + other


def set_seed(seed: int = 42, dependencies: Optional[List[list | tuple | int]] = None):
    # print(f"[DEBUG] Setting seed: {seed}")
    if not ConfigParams.DETERMINISM:
        return
    if dependencies:
        seed = compute_seed(dependencies)
    # MAX_SEED = 2**32 - 1
    # seed %= MAX_SEED + 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seq_tostr(seq: Tensor | List) -> str:
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
