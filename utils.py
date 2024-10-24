import random
import time
from typing import Any, Callable, Generator, List

import numpy as np
import torch
from config import DEBUG


def printd(statement, level=1):
    """
    Prints the statement only if the specified level is lower than the debug 
    level.
    """
    if DEBUG and level <= DEBUG:
        print(statement)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimedGenerator:
    """
    A wrapper class for a generator that measures and stores the time taken 
    to yield each item from the generator.

    Attributes:
        generator (Generator): The original generator to be wrapped.
        times (List[float]): A list storing the time taken to yield each item.

    Methods:
        __iter__(): Yields the same items as the original generator, while 
                    measuring the time taken for each yield.
        get_times(): Returns the list of times taken for each yield operation.
    """

    def __init__(self, generator: Generator):
        """
        Initializes the TimedGenerator with the original generator.

        Args:
            generator (Generator): The generator to be wrapped and timed.
        """
        self.generator = generator
        self.times: List[float] = []

    def __iter__(self):
        while True:
            try:
                # Start the timer before fetching the next dataset
                start_time = time.time()

                # Retrieve the next dataset using next()
                dataset = next(self.generator)

                # Stop the timer after getting the dataset
                elapsed_time = time.time() - start_time
                self.times.append(elapsed_time)

                # Yield the dataset
                yield dataset
            except StopIteration:
                # Exit the loop when the generator is exhausted
                break

    def get_times(self) -> List[float]:
        """
        Returns the list of times taken for each yield operation.

        Returns:
            List[float]: A list of elapsed times for each yield in seconds.
        """
        return self.times


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
