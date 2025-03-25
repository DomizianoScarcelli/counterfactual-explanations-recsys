from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

from recbole.config import Config
from recbole.trainer import Interaction
from torch import Tensor

from config.config import ConfigParams
from core.generation.dataset.generate import generate
from core.generation.dataset.utils import (
    get_dataloaders,
    interaction_to_tensor,
    load_dataset,
    save_dataset,
)
from core.generation.mutations import parse_mutations
from core.generation.strategies.abstract_strategy import GenerationStrategy
from core.generation.strategies.exhaustive import ExhaustiveStrategy
from core.generation.strategies.genetic import GeneticStrategy
from core.generation.strategies.genetic_categorized import CategorizedGeneticStrategy
from core.generation.strategies.targeted_uncategorized import (
    TargetedUncategorizedGeneticStrategy,
)
from core.generation.strategies.targeted import TargetedGeneticStrategy
from core.generation.utils import get_items
from core.models.config_utils import generate_model, get_config
from core.models.model_funcs import model_predict
from type_hints import GoodBadDataset
from utils.Split import Split


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

    def __init__(
        self,
        config: Optional[Config] = None,
        split: str = "test",
        whole_interaction: bool = False,
    ):
        super().__init__()
        self.config = (
            config if config else get_config(ConfigParams.DATASET, ConfigParams.MODEL)
        )
        self.whole_interaction = whole_interaction

        train_data, eval_data, test_data = get_dataloaders(self.config)

        if split == "train":
            self.data = train_data  # [1,2,3] -> 4
        elif split == "test":
            self.data = test_data  # [1,2,3] -> 4
        elif split == "eval":
            self.data = eval_data  # [2,3,4] -> 5
        else:
            raise NotImplementedError(f"Split must be train, eval or test, not {split}")

        # NOTE: this may be memory inefficent for large datasets, since calling
        # list over a generator will allocate all the items in the memory. A
        # solution is to materialize the list in batches. Whenever the i is
        # bigger than the current list, then we load another batch.

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
            data = self.data[self.index]  # type: ignore
            self.index += 1
            if self.whole_interaction:
                return data
            # the actual sequence is the first element of the tuple
            interaction = data[0]
            return interaction
        except IndexError:
            raise StopIteration

    def reset(self):
        self.index = 0


class SequenceGenerator(InteractionGenerator):
    def __init__(self, config: Config):
        self.config = config
        super().__init__(config)

    def next(self) -> Tensor:  # type: ignore
        interaction = super().next()
        return interaction_to_tensor(interaction)


class DatasetGenerator(SkippableGenerator):
    def __init__(
        self,
        config: Optional[Config] = None,
        limit_generation_to: Optional[Literal["good", "bad"]] = None,
        genetic_split: Optional[Split] = None,
        target: Optional[str | int] = None,
        use_cache: bool = False,
        return_interaction: bool = False,
        alphabet: Optional[List[int]] = None,
    ):
        super().__init__()
        self.config = (
            config if config else get_config(ConfigParams.DATASET, ConfigParams.MODEL)
        )
        self.interactions = InteractionGenerator(self.config)
        self.model = generate_model(self.config)
        self.use_cache = use_cache
        self.return_interaction = return_interaction
        self.strategy = ConfigParams.GENERATION_STRATEGY
        self.alphabet = alphabet if alphabet else list(get_items())
        self.target = target
        self.limit_generation_to = limit_generation_to
        self.genetic_split = (
            genetic_split if not ConfigParams.IGNORE_GEN_SPLIT else None
        )

    def skip(self):
        super().skip()
        self.interactions.skip()

    def instantiate_strategy(
        self, seq
    ) -> Tuple[Optional[GenerationStrategy], Optional[GenerationStrategy]]:
        if self.strategy == "genetic":
            sequence = seq.squeeze(0)
            assert len(sequence.shape) == 1, f"Sequence dim must be 1: {sequence.shape}"
            allowed_mutations = parse_mutations(ConfigParams.ALLOWED_MUTATIONS)
            self.good_strat = GeneticStrategy(
                input_seq=sequence,
                model=lambda x: model_predict(seq=x, model=self.model, prob=True),
                allowed_mutations=allowed_mutations,
                pop_size=ConfigParams.POP_SIZE,
                good_examples=True,
                generations=ConfigParams.GENERATIONS,
                halloffame_ratio=ConfigParams.HALLOFFAME_RATIO,
                alphabet=self.alphabet,
                split=self.genetic_split,
            )
            self.bad_strat = GeneticStrategy(
                input_seq=sequence,
                model=lambda x: model_predict(seq=x, model=self.model, prob=True),
                allowed_mutations=allowed_mutations,
                pop_size=ConfigParams.POP_SIZE,
                good_examples=False,
                generations=ConfigParams.GENERATIONS,
                halloffame_ratio=ConfigParams.HALLOFFAME_RATIO,
                alphabet=self.alphabet,
                split=self.genetic_split,
            )

        elif self.strategy == "genetic_categorized":
            sequence = seq.squeeze(0)
            assert len(sequence.shape) == 1, f"Sequence dim must be 1: {sequence.shape}"
            allowed_mutations = parse_mutations(ConfigParams.ALLOWED_MUTATIONS)

            self.good_strat = CategorizedGeneticStrategy(
                input_seq=sequence,
                model=lambda x: model_predict(seq=x, model=self.model, prob=True),
                allowed_mutations=allowed_mutations,
                pop_size=ConfigParams.POP_SIZE,
                good_examples=True,
                generations=ConfigParams.GENERATIONS,
                halloffame_ratio=ConfigParams.HALLOFFAME_RATIO,
                alphabet=self.alphabet,
                split=self.genetic_split,
            )
            self.bad_strat = CategorizedGeneticStrategy(
                input_seq=sequence,
                model=lambda x: model_predict(seq=x, model=self.model, prob=True),
                allowed_mutations=allowed_mutations,
                pop_size=ConfigParams.POP_SIZE,
                good_examples=False,
                generations=ConfigParams.GENERATIONS,
                halloffame_ratio=ConfigParams.HALLOFFAME_RATIO,
                alphabet=self.alphabet,
                split=self.genetic_split,
            )
        elif self.strategy == "targeted":
            if self.target is None:
                raise ValueError("target must not be None if strategy is 'targeted'")

            assert isinstance(self.target, str)
            sequence = seq.squeeze(0)
            assert len(sequence.shape) == 1, f"Sequence dim must be 1: {sequence.shape}"
            allowed_mutations = parse_mutations(ConfigParams.ALLOWED_MUTATIONS)

            self.good_strat = TargetedGeneticStrategy(
                input_seq=sequence,
                target=self.target,
                model=lambda x: model_predict(seq=x, model=self.model, prob=True),
                allowed_mutations=allowed_mutations,
                pop_size=ConfigParams.POP_SIZE,
                good_examples=False,  # inverted on purpose
                generations=ConfigParams.GENERATIONS,
                halloffame_ratio=ConfigParams.HALLOFFAME_RATIO,
                alphabet=self.alphabet,
                split=self.genetic_split,
            )
            self.bad_strat = TargetedGeneticStrategy(
                input_seq=sequence,
                target=self.target,
                model=lambda x: model_predict(seq=x, model=self.model, prob=True),
                allowed_mutations=allowed_mutations,
                pop_size=ConfigParams.POP_SIZE,
                good_examples=True,  # inverted on purpose
                generations=ConfigParams.GENERATIONS,
                halloffame_ratio=ConfigParams.HALLOFFAME_RATIO,
                alphabet=self.alphabet,
                split=self.genetic_split,
            )

        elif self.strategy == "targeted_uncategorized":
            if self.target is None:
                raise ValueError("target must not be None if strategy is 'targeted'")

            assert isinstance(self.target, int)
            sequence = seq.squeeze(0)
            assert len(sequence.shape) == 1, f"Sequence dim must be 1: {sequence.shape}"
            allowed_mutations = parse_mutations(ConfigParams.ALLOWED_MUTATIONS)

            self.good_strat = TargetedUncategorizedGeneticStrategy(
                input_seq=sequence,
                target=self.target,
                model=lambda x: model_predict(seq=x, model=self.model, prob=True),
                allowed_mutations=allowed_mutations,
                pop_size=ConfigParams.POP_SIZE,
                good_examples=False,  # inverted on purpose
                generations=ConfigParams.GENERATIONS,
                halloffame_ratio=ConfigParams.HALLOFFAME_RATIO,
                alphabet=self.alphabet,
                split=self.genetic_split,
            )
            self.bad_strat = TargetedUncategorizedGeneticStrategy(
                input_seq=sequence,
                target=self.target,
                model=lambda x: model_predict(seq=x, model=self.model, prob=True),
                allowed_mutations=allowed_mutations,
                pop_size=ConfigParams.POP_SIZE,
                good_examples=True,  # inverted on purpose
                generations=ConfigParams.GENERATIONS,
                halloffame_ratio=ConfigParams.HALLOFFAME_RATIO,
                alphabet=self.alphabet,
                split=self.genetic_split,
            )

        elif self.strategy == "brute_force":
            self.good_strat = ExhaustiveStrategy(
                input_seq=seq,
                model=self.model,
                alphabet=self.alphabet,
                good_examples=True,
            )
            self.bad_strat = ExhaustiveStrategy(
                input_seq=seq,
                model=self.model,
                alphabet=self.alphabet,
                good_examples=False,
            )
        else:
            raise NotImplementedError(
                f"Generations strategy '{self.strategy}' not implemented, choose between 'generation' and 'exhaustive'"
            )
        if self.limit_generation_to == "bad":
            self.good_strat = None
        if self.limit_generation_to == "good":
            self.bad_strat = None
        return self.good_strat, self.bad_strat

    def next(self) -> GoodBadDataset | Tuple[GoodBadDataset, Interaction]:
        assert (
            self.interactions.index == self.index
        ), f"{self.interactions.index} != {self.index} at the start of the method"
        interaction = self.interactions.next()
        cache_path = Path(f".dataset_cache/interaction_{self.index}_dataset.pickle")
        # TODO: make cache path aware of the strategy
        if cache_path.exists() and self.use_cache:
            if self.strategy != "generation":
                raise NotImplementedError(
                    "Cache not implemented for dataset not generated with the 'generation' strategy"
                )
            dataset = load_dataset(cache_path)
        else:
            good_strat, bad_strat = self.instantiate_strategy(
                interaction_to_tensor(interaction)
            )
            dataset = generate(
                interaction=interaction, good_strat=good_strat, bad_strat=bad_strat
            )
            if self.use_cache:
                save_dataset(dataset, cache_path)
        self.index += 1
        assert (
            self.interactions.index == self.index
        ), f"{self.interactions.index} != {self.index} at the end of the method"
        if self.return_interaction:
            return dataset, interaction
        return dataset

    def __iter__(self) -> DatasetGenerator:
        self.reset()
        return self

    def match_indices(self):
        # Because of the EmptyDatasetError, there may be a mismatch between the dataset indices.
        if self.interactions.index != self.index:
            _min = min(self.interactions.index, self.index)
            self.interactions.index = _min
            self.index = _min

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
        assert isinstance(
            generator, SkippableGenerator
        ), f"Generator of type {type(generator)} not supported"
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


if __name__ == "__main__":
    confg = get_config(model=ConfigParams.MODEL, dataset=ConfigParams.DATASET)
    ints = InteractionGenerator(confg)
    count = 0
    for i in ints:
        count += 1
    print(f"There are {count} sequences")
