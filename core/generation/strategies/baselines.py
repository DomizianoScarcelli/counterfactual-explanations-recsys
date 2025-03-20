from utils.generators import DatasetGenerator, InteractionGenerator, SequenceGenerator
from utils.RunLogger import RunLogger
from core.models.utils import topk
from utils.distances import edit_distance
from core.generation.utils import equal_ys
from typing import Tuple
from core.generation.utils import get_items
from core.models.config_utils import generate_model, get_config
from typing import Optional
from config.config import ConfigParams
from utils.Split import Split
from typing import Callable, List
import random

import torch
from torch import Tensor

from core.generation.strategies.abstract_strategy import GenerationStrategy
from core.models.utils import trim
from utils.utils import seq_tostr


BASELINES_DB_PATH = "results/evaluate/baselines.db"


class RandomStrategy(GenerationStrategy):
    def __init__(
        self,
        input_seq: Tensor,
        model: Callable,
        split: Split,
        alphabet: List[int],
        num_edits: int,
        ks: Optional[List[int]] = None,
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
        self.alphabet = alphabet
        self.gt = model(input_seq)
        self.split = split
        self.num_edits = num_edits
        self.ks = ConfigParams.TOPK if not ks else ks
        self.logger = RunLogger(db_path=BASELINES_DB_PATH, add_config=True)

    def generate(self) -> List[Tuple[List[int], Tensor]]:
        x = trim(self.input_seq.squeeze(0)).tolist()
        x_primes = []
        for _ in range(ConfigParams.POP_SIZE):
            executed, mutable, fixed = self.split.apply(x)
            random_idxs = random.sample(range(len(mutable)), k=self.num_edits)
            random_chars = random.sample(self.alphabet, k=self.num_edits)
            for idx, char in zip(random_idxs, random_chars):
                mutable[idx] = char
            new_seq = executed + mutable + fixed
            logits = self.model(torch.tensor(new_seq).unsqueeze(0))
            x_primes.append((new_seq, logits))
        return x_primes

    def log(self, dataset):
        #TODO: generate one log for each of the four different settings
        rankings = {k: topk(logits=self.gt, dim=-1, k=k, indices=True) for k in self.ks}
        for x, y in dataset:
            crankings = {
                k: topk(
                    logits=y,
                    dim=-1,
                    k=k,
                    indices=True,
                )
                for k in self.ks
            }
            # print({k: r.squeeze() for k, r in crankings.items()})
            scores = {k: equal_ys(crankings[k].squeeze(), rankings[k].squeeze(), return_score=True)[1] for k in self.ks}  # type: ignore
            distance = edit_distance(x, self.input_seq, normalized=False)
            log = {
                "split": self.split,
                "source": seq_tostr(self.input_seq.squeeze(0)),
                "aligned": seq_tostr(x),
                "cost": distance,
            }

            log_at_ks = [
                {
                    f"aligned_gt@{k}": seq_tostr(crankings[k].squeeze(0)),
                    f"gt@{k}": seq_tostr(rankings[k].squeeze(0)),
                    f"score@{k}": scores[k],
                }
                for k in self.ks
            ]
            for log_at_k in log_at_ks:
                log = {**log, **log_at_k}
            self.logger.log_run(log, primary_key=["source", "split", "aligned"], tostr=True)


class EducatedStategy(GenerationStrategy):
    pass


class PopularStrategy(GenerationStrategy):
    pass


if __name__ == "__main__":
    conf = get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)
    model = generate_model(config=conf)
    seqs = SequenceGenerator(conf)
    for seq in seqs:
        strat = RandomStrategy(
            input_seq=seq,
            model=model,
            split=Split(None, 10, None),
            alphabet=list(get_items()),
            num_edits=1,
        )
        pop = strat.generate()
        strat.log(pop)
        print(f"Logged")
