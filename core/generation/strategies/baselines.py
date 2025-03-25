from abc import abstractmethod
from utils.generators import SequenceGenerator
from utils.RunLogger import RunLogger
from core.models.utils import topk
from utils.distances import edit_distance
from core.generation.utils import equal_ys, labels2cat
from tqdm import tqdm
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


class BaselineStrategy(GenerationStrategy):
    def __init__(
        self,
        input_seq: Tensor,
        model: Callable,
        split: Split,
        alphabet: List[int],
        num_edits: int,
        target_cats: List[int],
        target_items: List[int],
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
        self.target_cats = target_cats
        self.target_items = target_items
        self.logger = RunLogger(db_path=BASELINES_DB_PATH, add_config=True)

    @abstractmethod
    def generate(self) -> List[Tuple[List[int], Tensor]]:
        pass

    def log(self, dataset):
        print("Generating and logging results...")
        uncat_rankings = {
            k: [
                {x.item()}
                for x in topk(logits=self.gt, dim=-1, k=k, indices=True).squeeze(0)
            ]
            for k in self.ks
        }
        cat_rankings = {
            k: labels2cat(
                topk(logits=self.gt, dim=-1, k=k, indices=True).squeeze(0), encode=True
            )
            for k in self.ks
        }

        for x, y in dataset:
            cat_counter_rankings = {
                k: [
                    {x.item()}
                    for x in topk(
                        logits=y,
                        dim=-1,
                        k=k,
                        indices=True,
                    ).squeeze(0)
                ]
                for k in self.ks
            }
            uncat_counter_rankings = {
                k: labels2cat(
                    topk(
                        logits=y,
                        dim=-1,
                        k=k,
                        indices=True,
                    ).squeeze(0),
                    encode=True,
                )
                for k in self.ks
            }
            uncat_scores = {k: equal_ys(cat_counter_rankings[k], uncat_rankings[k], return_score=True)[1] for k in self.ks}  # type: ignore
            cat_scores = {k: equal_ys(uncat_counter_rankings[k], cat_rankings[k], return_score=True)[1] for k in self.ks}  # type: ignore
            uncat_target_scores = {}
            cat_target_scores = {}
            for target_item in self.target_items:
                uncat_target_gt = {
                    k: [{target_item} for _ in range(k)] for k in self.ks
                }
                targ_uncat_scores = {k: equal_ys(uncat_target_gt[k], uncat_rankings[k], return_score=True)[1] for k in self.ks}  # type: ignore
                uncat_target_scores[target_item] = targ_uncat_scores

            for target_cat in self.target_cats:
                cat_target_gt = {k: [{target_cat} for _ in range(k)] for k in self.ks}
                targ_cat_scores = {k: equal_ys(cat_target_gt[k], cat_rankings[k], return_score=True)[1] for k in self.ks}  # type: ignore
                cat_target_scores[target_cat] = targ_cat_scores

            x = torch.tensor(x)
            trimmed_seq = trim(self.input_seq.squeeze(0))
            distance = edit_distance(x, trimmed_seq, normalized=False)
            log = {
                "split": self.split,
                "source": seq_tostr(trimmed_seq),
                "aligned": seq_tostr(x),
                "cost": distance,
            }
            cat_log_at_ks = [
                {
                    f"rakings@{k}": seq_tostr(cat_rankings[k]),
                    f"counter_rankings@{k}": seq_tostr(cat_counter_rankings[k]),
                    f"score@{k}": cat_scores[k],
                }
                for k in self.ks
            ]
            cat_log = {}
            for cat_log_at_k in cat_log_at_ks:
                cat_log = {
                    **log,
                    **cat_log_at_k,
                    "categorized": True,
                    "targeted": False,
                }
            self.logger.log_run(
                cat_log, primary_key=["source", "split", "aligned"], tostr=True
            )
            uncat_log_at_ks = [
                {
                    f"rakings@{k}": seq_tostr(uncat_rankings[k]),
                    f"counter_rankings@{k}": seq_tostr(uncat_counter_rankings[k]),
                    f"score@{k}": uncat_scores[k],
                }
                for k in self.ks
            ]
            uncat_log = {}
            for uncat_log_at_k in uncat_log_at_ks:
                uncat_log = {
                    **log,
                    **uncat_log_at_k,
                    "categorized": False,
                    "targeted": False,
                }
            self.logger.log_run(
                uncat_log, primary_key=["source", "split", "aligned"], tostr=True
            )

            for target_item, target_uncat_score in uncat_target_scores.items():
                uncat_log = {}
                targ_uncat_log_at_ks = [
                    {
                        f"rakings@{k}": seq_tostr(uncat_target_gt[k]),
                        f"counter_rankings@{k}": seq_tostr(uncat_counter_rankings[k]),
                        f"score@{k}": target_uncat_score,
                    }
                    for k in self.ks
                ]
                for targ_uncat_log in targ_uncat_log_at_ks:
                    uncat_log = {
                        **log,
                        **targ_uncat_log,
                        "categorized": False,
                        "targeted": True,
                        "target": target_item,
                    }
                self.logger.log_run(
                    uncat_log, primary_key=["source", "split", "aligned"], tostr=True
                )
            for target_cat, target_cat_score in cat_target_scores.items():
                cat_log = {}
                targ_cat_log_at_ks = [
                    {
                        f"rakings@{k}": seq_tostr(cat_target_gt[k]),
                        f"counter_rankings@{k}": seq_tostr(cat_counter_rankings[k]),
                        f"score@{k}": target_cat_score,
                    }
                    for k in self.ks
                ]
                for targ_cat_log in targ_cat_log_at_ks:
                    cat_log = {
                        **log,
                        **targ_cat_log,
                        "categorized": True,
                        "targeted": True,
                        "target": target_cat,
                    }
                self.logger.log_run(
                    cat_log, primary_key=["source", "split", "aligned"], tostr=True
                )


class RandomStrategy(BaselineStrategy):
    def __init__(
        self,
        input_seq: Tensor,
        model: Callable,
        split: Split,
        alphabet: List[int],
        num_edits: int,
        target_cats: List[int],
        target_items: List[int],
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
            num_edits=num_edits,
            target_items=target_items,
            target_cats=target_cats,
            split=split,
            ks=ks,
        )

    def generate(self) -> List[Tuple[List[int], Tensor]]:
        x = trim(self.input_seq.squeeze(0)).tolist()
        x_primes = []
        for _ in tqdm(
            range(ConfigParams.POP_SIZE),
            desc="Generating dataset with RandomStrategy...",
        ):
            executed, mutable, fixed = self.split.apply(x)
            random_idxs = random.sample(range(len(mutable)), k=self.num_edits)
            random_chars = random.sample(self.alphabet, k=self.num_edits)
            for idx, char in zip(random_idxs, random_chars):
                mutable[idx] = char
            new_seq = executed + mutable + fixed
            logits = self.model(torch.tensor(new_seq).unsqueeze(0))
            x_primes.append((new_seq, logits))
        return x_primes


class EducatedStategy(GenerationStrategy):
    pass


class PopularStrategy(GenerationStrategy):
    # TODO: skip this for now, implement it later.
    pass


if __name__ == "__main__":
    conf = get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)
    model = generate_model(config=conf)
    seqs = SequenceGenerator(conf)
    target_cats = ["Horror", "Action", "Adventure", "Animation", "Fantasy", "Drama"]
    target_items = [50, 411, 630, 1305]
    for seq in seqs:
        strat = RandomStrategy(
            input_seq=seq,
            model=model,
            split=Split(None, 10, None),
            alphabet=list(get_items()),
            num_edits=1,
            target_cats=target_cats,
            target_items=target_items,
        )
        pop = strat.generate()
        strat.log(pop)
        print(f"Logged")
        break
