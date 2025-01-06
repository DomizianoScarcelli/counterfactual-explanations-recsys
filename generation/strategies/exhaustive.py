from typing import Callable, List

import torch
from torch import Tensor

from generation.strategies.abstract_strategy import GenerationStrategy
from models.utils import trim
from type_hints import Dataset


class ExhaustiveStrategy(GenerationStrategy):
    def __init__(
        self,
        input_seq: Tensor,
        model: Callable,
        alphabet: List[int],
        good_examples: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            input_seq=input_seq,
            model=model,
            alphabet=alphabet,
            good_examples=good_examples,
            verbose=verbose,
        )
        self.alphabet = torch.tensor(alphabet)
        self.gt = model(input_seq).argmax(-1).item()
        diff_positions = min(5, input_seq.size(-1) - 1)
        self.positions_list = [
            input_seq.size(-1) - 1 - i for i in range(diff_positions)
        ]

    def generate(self) -> Dataset:
        x = trim(self.input_seq.squeeze(0)).unsqueeze(0)
        x_primes = x.repeat(len(self.alphabet), 1)

        assert x_primes.shape == torch.Size(
            [len(self.alphabet), x.size(1)]
        ), f"x shape uncorrect: {x_primes.shape} != {[len(self.alphabet), x.size(1)]}"
        result_xs = torch.tensor([])
        result_out = torch.tensor([])
        for position in self.positions_list:
            positions = torch.tensor([position] * len(self.alphabet))

            x_primes[torch.arange(len(self.alphabet)), positions] = self.alphabet
            out_primes = self.model(x_primes).argmax(-1)

            diff_mask = out_primes != self.gt
            eq_mask = out_primes == self.gt

            diff_x_primes = x_primes[diff_mask]
            diff_out_primes = out_primes[diff_mask]
            equal_x_primes = x_primes[eq_mask]
            equal_out_primes = out_primes[eq_mask]

            if self.good_examples:
                result_xs = torch.cat((result_xs, equal_x_primes))
                result_out = torch.cat((result_out, equal_out_primes))
            else:
                result_xs = torch.cat((result_xs, diff_x_primes))
                result_out = torch.cat((result_out, diff_out_primes))

        return [
            (x.to(torch.int16), int(label.item()))
            for x, label in zip(result_xs, result_out)
        ]

    def replace_alphabet(self, alphabet):
        self.alphabet = torch.tensor(alphabet)


# if __name__ == "__main__":
#     conf = get_config(ConfigParams.DATASET, ConfigParams.MODEL)
#     model = generate_model(conf)
#     sequences = SequenceGenerator(conf)
#     seq = next(sequences)
#     alphabet = list(get_items())
#     strat = ExhaustiveGenerationStrategy(
#             input_seq = seq,
#             model = model,
#             alphabet = alphabet,
#             position=seq.size(1)-1)
#     strat.generate()
