from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from recbole.config import Config
from recbole.trainer import Interaction
from torch import Tensor

from genetic.dataset.generate import generate
from genetic.dataset.utils import (get_dataloaders,
                                   interaction_to_tensor, load_dataset,
                                   save_dataset)
from models.config_utils import generate_model
from type_hints import GoodBadDataset


class SkippableGenerator(ABC):
    """
    Abstract base class for a skippable generator. Provides a mechanism for generating
        elements, skipping elements, and resetting the generator state.

        Attributes:
            index (int): The current position of the generator.
    """
    def __init__(self):
        self.index = 0
        
    def skip(self) -> None:
        """
        Skips to the next element without performing the generation.
        """
        self.index += 1
        pass

    def peek(self) -> Any:
        """
        Returns the next generation without incrementing the index
        """
        result = next(self)
        self.index -= 1
        return result

    @abstractmethod
    def next(self) -> Any:
        """
        Generates the next element.

        Returns:
            The next generated element.
        """
        pass
    
    def __iter__(self) -> SkippableGenerator:
        """
        Resets the generator index and returns itself for iteration.

        Returns:
            The current generator instance.
        """
        self.index = 0
        return self

    def __next__(self) -> Any:
        """
        Calls the next() method for iteration.

        Returns:
            The next generated element.

        Raises:
            StopIteration: If no more elements are available.
        """
        return self.next()
    
    @abstractmethod
    def reset(self) -> None:
        """
        Resets the generator to its initial state.
        """
        pass


class InteractionGenerator(SkippableGenerator):
    """
    A generator for iterating over interactions from a dataset.

    Attributes:
        config (Config): Configuration object containing generator settings.
        data (list): List of data items.
    """
    def __init__(self, config: Config, split: str = "test"):
        super().__init__()
        self.config = config
        train_data, eval_data, test_data = get_dataloaders(config)
        
        if split == "train":
            self.data = train_data
        elif split == "test":
            self.data = test_data
        elif split == "eval":
            self.data = eval_data
        else:
            raise NotImplementedError(f"Split must be train, eval or test, not {split}")



        #NOTE: this may be memory inefficent for large datasets, since calling
        #list over a generator will allocate all the items in the memory. A
        #solution is to materialize the list in batches. Whenever the i is
        #bigger than the current list, then we load another batch.
        
        self.data = list(self.data)
        # Self.data is a List of Tuples of the form:
        # (Interaction, None [1], batch_idx [B], ground_truth(?) [B])

        # Example of Interaction.interaction structure
        # {
        #         'user_id': Tensor,  # Shape: [B]
        #         'item_id': Tensor,  # Shape: [B]
        #         'rating': Tensor,  # Shape: [B]
        #         'timestamp': Tensor,  # Shape: [B]
        #         'item_length': Tensor,  # Shape: [B]
        #         'item_id_list': Tensor,  # Shape: [B, MAX_LENGTH]
        #         'rating_list': Tensor,  # Shape: [B, MAX_LENGTH]
        #         'timestamp_list': Tensor,  # Shape: [B, MAX_LENGTH]
        #         'Mask_item_id_list': Tensor,  # Shape: [B, MAX_LENGTH]
        #         'Pos_item_id': Tensor,  # Shape: [B, n_masks]
        #         'Neg_item_id': Tensor,  # Shape: [B, n_masks]
        #         'MASK_INDEX': Tensor,  # Shape: [B, n_masks]
        # }

        # for i, elem in enumerate(self.data):
        #     print(f"Element {i} is made of: ")
        #     for j, subel in enumerate(elem):
        #         print(f"    Subelement {j} is: {subel} of type {type(subel)}")
        # print(f"Interaction is: {self.data[0][0].interaction}")

    def next(self) -> Interaction:
        try:
            data = self.data[self.index] #type: ignore
            # the actual sequence is the first element of the tuple
            interaction = data[0]
            self.index += 1
            return interaction
        except IndexError:
            raise StopIteration

    def reset(self):
        self.index = 0


class SequenceGenerator(InteractionGenerator):
    def __init__(self, config: Config):
        super().__init__(config)

    def next(self) -> Tensor: #type: ignore
        interaction = super().next()
        return interaction_to_tensor(interaction)

class DatasetGenerator(SkippableGenerator):
    def __init__(self, config: Config, use_cache: bool=True, return_interaction: bool=False):
        super().__init__()
        self.config = config
        self.interactions = InteractionGenerator(config)
        self.model = generate_model(config)
        self.use_cache = use_cache
        self.return_interaction = return_interaction

    def skip(self):
        super().skip()
        self.interactions.skip()

    def next(self) -> GoodBadDataset | Tuple[GoodBadDataset, Interaction]:
        assert self.interactions.index == self.index, f"{self.interactions.index} != {self.index} at the start of the method"
        interaction = self.interactions.next()
        cache_path = os.path.join(f"dataset_cache/interaction_{self.index}_dataset.pickle")
        if os.path.exists(cache_path) and self.use_cache:
            dataset = load_dataset(cache_path)
        else:
            dataset = generate(interaction, self.model)
            if self.use_cache:
                save_dataset(dataset, cache_path)
        self.index += 1
        assert self.interactions.index == self.index, f"{self.interactions.index} != {self.index} at the end of the method"
        if self.return_interaction:
            return dataset, interaction
        return dataset

    def __iter__(self) -> DatasetGenerator:
        self.reset()
        return self

    def reset(self):
        self.interactions.reset()
        self.index = 0

    def stats(self) -> Optional[Tuple[float, float]]:
        pass


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


    def __init__(self, generator: SkippableGenerator):
        """
        Initializes the TimedGenerator with the original generator.

        Args:
            generator (Generator): The generator to be wrapped and timed.
        """
        assert isinstance(generator, SkippableGenerator), f"Generator of type {type(generator)} not supported"
        self.generator = generator
        self.times: List[Optional[float]] = []

    def __next__(self):
        start_time = time.time()
        try:
            item = next(self.generator)
        except StopIteration:
            raise
        elapsed_time = time.time() - start_time
        self.times.append(elapsed_time)
        return item

    def skip(self):
        self.times.append(None)
        return self.generator.skip()

    @property
    def index(self):
        return self.generator.index

    def __iter__(self):
        return self

    def get_times(self) -> List[Optional[float]]:
        """
        Returns the list of times taken for each yield operation.

        Returns:
            A list of elapsed times for each yield in seconds. If a generation
            is skipped, the time will be None.
        """
        return self.times

