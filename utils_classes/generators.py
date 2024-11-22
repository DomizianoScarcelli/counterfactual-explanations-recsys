from __future__ import annotations
from abc import ABC, abstractmethod
from genetic.dataset.utils import get_dataloaders, get_sequence_from_interaction, save_dataset

from recbole.config import Config
from recbole.trainer import Interaction
from typing import Any, Tuple
from torch import Tensor
import os
from models.config_utils import generate_model
from genetic.dataset.utils import load_dataset, save_dataset
from genetic.dataset.generate import generate
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
        test_data (list): List of test data items.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        _, _, self.test_data = get_dataloaders(config)
        assert self.test_data is not None, "Test data is None"

        #NOTE: this may be memory inefficent for large datasets, since calling
        #list over a generator will allocate all the items in the memory. A
        #solution is to materialize the list in batches. Whenever the i is
        #bigger than the current list, then we load another batch.

        self.test_data = list(self.test_data)

    def next(self) -> Interaction:
        try:
            data = self.test_data[self.index] #type: ignore
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
        return get_sequence_from_interaction(interaction)

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
