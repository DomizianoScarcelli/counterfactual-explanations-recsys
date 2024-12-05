from typing import Callable, List

import torch
from torch import Tensor

from genetic.abstract_generation import GenerationStrategy
from models.utils import trim
from type_hints import Dataset


class ExhaustiveGenerationStrategy(GenerationStrategy):
    def __init__(self, 
                 input_seq: Tensor, 
                 model: Callable,
                 alphabet: List[int],
                 position: int,
                 good_examples: bool=True,
                 verbose: bool=True):
        self.input_seq = input_seq
        self.model = model
        self.alphabet = torch.tensor(alphabet)
        self.position = position
        self.good_examples = good_examples
        self.gt = model(input_seq).argmax(-1).item()

    def generate(self) -> Dataset:
        x = trim(self.input_seq.squeeze(0)).unsqueeze(0)
        x_primes = x.repeat(len(self.alphabet), 1)

        assert x_primes.shape == torch.Size([len(self.alphabet), x.size(1)]), f"x shape uncorrect: {x_primes.shape} != {[len(self.alphabet), x.size(1)]}"
        positions = torch.tensor([self.position] * len(self.alphabet))

        x_primes[torch.arange(len(self.alphabet)), positions] = self.alphabet
        out_primes = self.model(x_primes).argmax(-1)

        diff_mask = out_primes != self.gt
        eq_mask = out_primes == self.gt

        differing_x_primes = x_primes[diff_mask]
        differing_out_primes = out_primes[diff_mask]
        equal_x_primes = x_primes[eq_mask]

        print(differing_x_primes.shape)
        print(equal_x_primes.shape)

        if self.good_examples:
            return [(x, self.gt) for x in equal_x_primes]
        return [(x, label) for x, label in zip(differing_x_primes, differing_out_primes)]


    def clean(self, examples: Dataset) -> Dataset:
        pass

    def postprocess(self, population: Dataset) -> Dataset:
        pass

# if __name__ == "__main__":
#     conf = get_config(ConfigParams.DATASET, ConfigParams.MODEL)
#     model = generate_model(conf)
#     sequences = SequenceGenerator(conf)
#     seq = next(sequences)
#     alphabet = list(get_items(Items.ML_1M))
#     strat = ExhaustiveGenerationStrategy(
#             input_seq = seq,
#             model = model,
#             alphabet = alphabet,
#             position=seq.size(1)-1)
#     strat.generate()
