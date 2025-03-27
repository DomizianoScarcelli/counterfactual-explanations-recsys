from config.constants import cat2id
from abc import abstractmethod
from type_hints import RecDataset
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
import fire

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
        target_cats: List[str],
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

    def log(self, i: int, dataset, baseline_name: str):
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
        cat_logs = []
        uncat_logs = []
        targ_cat_logs = []
        targ_uncat_logs = []
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
                cat_target_gt = {
                    k: [{cat2id[target_cat]} for _ in range(k)] for k in self.ks
                }
                targ_cat_scores = {k: equal_ys(cat_target_gt[k], cat_rankings[k], return_score=True)[1] for k in self.ks}  # type: ignore
                cat_target_scores[target_cat] = targ_cat_scores

            x = torch.tensor(x)
            trimmed_seq = trim(self.input_seq.squeeze(0))
            distance = edit_distance(x, trimmed_seq, normalized=False)
            log = {
                "i": i,
                "split": self.split,
                "num_edits": self.num_edits,
                "source": seq_tostr(trimmed_seq),
                "aligned": seq_tostr(x),
                "cost": distance,
                "baseline": baseline_name,
            }
            cat_logs_k = [
                {
                    f"rankings@{k}": seq_tostr(cat_rankings[k]),
                    f"counter_rankings@{k}": seq_tostr(cat_counter_rankings[k]),
                    f"score@{k}": cat_scores[k],
                }
                for k in self.ks
            ]
            cat_log = {}
            for temp_log in cat_logs_k:
                cat_log.update(
                    {
                        **log,
                        **temp_log,
                        "categorized": True,
                        "targeted": False,
                    }
                )
            cat_logs.append(cat_log)

            uncat_log_ks = [
                {
                    f"rankings@{k}": seq_tostr(uncat_rankings[k]),
                    f"counter_rankings@{k}": seq_tostr(uncat_counter_rankings[k]),
                    f"score@{k}": uncat_scores[k],
                }
                for k in self.ks
            ]
            uncat_log = {}
            for temp_log in uncat_log_ks:
                uncat_log.update(
                    {
                        **log,
                        **temp_log,
                        "categorized": False,
                        "targeted": False,
                    }
                )
            uncat_logs.append(uncat_log)

            for target_item in uncat_target_scores:
                targ_uncat_log_ks = [
                    {
                        f"rankings@{k}": seq_tostr(uncat_target_gt[k]),
                        f"counter_rankings@{k}": seq_tostr(uncat_counter_rankings[k]),
                        f"score@{k}": uncat_target_scores[target_item][k],
                    }
                    for k in self.ks
                ]
                targ_uncat_log = {}
                for temp_log in targ_uncat_log_ks:
                    targ_uncat_log.update(
                        {
                            **log,
                            **temp_log,
                            "categorized": False,
                            "targeted": True,
                            "target": target_item,
                        }
                    )
                targ_uncat_logs.append(targ_uncat_log)
            for target_cat in cat_target_scores:
                targ_cat_log_ks = [
                    {
                        f"rankings@{k}": seq_tostr(cat_target_gt[k]),
                        f"counter_rankings@{k}": seq_tostr(cat_counter_rankings[k]),
                        f"score@{k}": cat_target_scores[target_cat][k],
                    }
                    for k in self.ks
                ]
                targ_cat_log = {}
                for temp_log in targ_cat_log_ks:
                    targ_cat_log.update(
                        {
                            **log,
                            **temp_log,
                            "categorized": True,
                            "targeted": True,
                            "target": target_cat,
                        }
                    )
                targ_cat_logs.append(targ_cat_log)

        # Take and log best counterfactuals from all logs
        best_cat_log = min(cat_logs, key=lambda x: x["score@1"])
        best_uncat_log = min(uncat_logs, key=lambda x: x["score@1"])
        best_targ_cat_logs = {
            t: min(
                [log for log in targ_cat_logs if log["target"] == t],
                key=lambda x: x["score@1"],
            )
            for t in {log["target"] for log in targ_cat_logs}
        }
        best_targ_uncat_logs = {
            t: min(
                [log for log in targ_uncat_logs if log["target"] == t],
                key=lambda x: x["score@1"],
            )
            for t in {log["target"] for log in targ_uncat_logs}
        }
        untarg_logs = [
            best_cat_log,
            best_uncat_log,
        ]
        targ_logs = [best_targ_cat_logs, best_targ_uncat_logs]
        for log in targ_logs:
            for _, target_log in log.items():
                self.logger.log_run(
                    target_log,
                    primary_key=["source", "split", "aligned", "target"],
                    tostr=True,
                )
        for log in untarg_logs:
            self.logger.log_run(
                log,
                primary_key=["source", "split", "aligned"],
                tostr=True,
            )


class RandomStrategy(BaselineStrategy):
    def __init__(
        self,
        input_seq: Tensor,
        model: Callable,
        split: Split,
        alphabet: List[int],
        num_edits: int,
        target_cats: List[str],
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
            leave=False,
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


dataset_target_items = {
        RecDataset.ML_100K:[50, 411, 630, 1305],
        RecDataset.STEAM:[50, 411, 630, 1305],
        RecDataset.ML_1M:[50, 411, 630, 1305],
        }

dataset_target_cats = {
        RecDataset.ML_100K:["Horror", "Action", "Adventure", "Animation", "Fantasy", "Drama"]
        RecDataset.STEAM:[50, 411, 630, 1305],
        RecDataset.ML_1M:["Horror", "Action", "Adventure", "Animation", "Fantasy", "Drama"]
        }

def run_random_baseline():
    conf = get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)
    model = generate_model(config=conf)
    seqs = SequenceGenerator(conf)
    target_cats = dataset_target_cats[ConfigParams.DATASET]
    target_items = dataset_target_items[ConfigParams.DATASET]
    total = 0
    for i in seqs:
        total += 1
    seqs.reset()
    pbar = tqdm(total=total, desc="Evaluating RandomStrategy baseline...")
    for i, seq in enumerate(seqs):
        strat = RandomStrategy(
            input_seq=seq,
            model=model,
            split=Split(None, 10, None),
            alphabet=list(get_items()),
            num_edits=1,
            target_cats=target_cats,
            target_items=target_items,
        )
        exists = strat.logger.exists(
            log={"i": i}, primary_key=["i"], consider_config=True
        )
        if exists:
            pbar.total -= 1
            seqs.skip()
            continue
        pop = strat.generate()
        strat.log(i, pop, baseline_name="random")
        pbar.update(1)


def main(baseline):
    if baseline == "random":
        run_random_baseline()
    else:
        raise ValueError(f"Baseline {baseline} not supported yet")


if __name__ == "__main__":
    fire.Fire(main)
