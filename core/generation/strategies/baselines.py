from core.generation.utils import token2id
from core.generation.utils import get_category_map, label2cat
from typing import Set
import json

from config.constants import id2cat
from type_hints import RecDataset
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
from enum import Enum

baselines_db_path = lambda edits: (
    f"results/evaluate/baselines{'_2edits' if edits == 2 else ''}.db"
)


class CounterfactualSetting(Enum):
    TARGETED_UNCATEGORIZED = (0,)
    TARGETED_CATEGORIZED = (1,)
    UNTARGETED_UNCATEGORIZED = (2,)
    UNTARGETED_CATEGORIZED = (3,)


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
        verbose: bool = True,
    ):
        super().__init__(
            input_seq=input_seq,
            model=model,
            alphabet=alphabet,
            verbose=verbose,
        )
        self.alphabet = alphabet
        self.gt = model(input_seq)
        self.split = split
        self.num_edits = num_edits
        self.ks = ConfigParams.TOPK if not ks else ks
        self.target_cats = target_cats
        self.target_items = target_items
        self.logger = RunLogger(
            db_path=baselines_db_path(self.num_edits), add_config=True
        )

    @abstractmethod
    def generate(self) -> List[Tuple[List[int], Tensor]]:
        pass

    def log(
        self,
        i: int,
        dataset,
        baseline_name: str,
        settings: Set[CounterfactualSetting] = set(CounterfactualSetting),
    ):
        uncat_rankings, cat_rankings = None, None

        cat_logs = []
        uncat_logs = []
        targ_cat_logs = []
        targ_uncat_logs = []
        if (
            CounterfactualSetting.UNTARGETED_UNCATEGORIZED in settings
            or CounterfactualSetting.TARGETED_UNCATEGORIZED in settings
        ):
            uncat_rankings = {
                k: [
                    {x.item()}
                    for x in topk(logits=self.gt, dim=-1, k=k, indices=True).squeeze(0)
                ]
                for k in self.ks
            }
        if (
            CounterfactualSetting.UNTARGETED_CATEGORIZED in settings
            or CounterfactualSetting.TARGETED_CATEGORIZED in settings
        ):
            cat_rankings = {
                k: labels2cat(
                    topk(logits=self.gt, dim=-1, k=k, indices=True).squeeze(0),
                    encode=True,
                )
                for k in self.ks
            }
        for x, y in dataset:
            cat_counter_rankings, uncat_counter_rankings = None, None
            if CounterfactualSetting.UNTARGETED_UNCATEGORIZED in settings:
                uncat_counter_rankings = {
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
                uncat_scores = {k: equal_ys(uncat_counter_rankings[k], uncat_rankings[k], return_score=True)[1] for k in self.ks}  # type: ignore

            if CounterfactualSetting.UNTARGETED_CATEGORIZED in settings:
                cat_counter_rankings = {
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
                cat_scores = {k: equal_ys(cat_counter_rankings[k], cat_rankings[k], return_score=True)[1] for k in self.ks}  # type: ignore

            if CounterfactualSetting.TARGETED_UNCATEGORIZED in settings:
                uncat_target_scores = {}
                for target_item in self.target_items:
                    uncat_target_gt = {
                        k: [{target_item} for _ in range(k)] for k in self.ks
                    }
                    targ_uncat_scores = {k: equal_ys(uncat_target_gt[k], uncat_rankings[k], return_score=True)[1] for k in self.ks}  # type: ignore
                    uncat_target_scores[target_item] = targ_uncat_scores

            if CounterfactualSetting.TARGETED_CATEGORIZED in settings:
                cat_target_scores = {}
                for target_cat in self.target_cats:
                    cat_target_gt = {
                        k: [{cat2id()[target_cat]} for _ in range(k)] for k in self.ks
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

            if CounterfactualSetting.UNTARGETED_CATEGORIZED in settings:
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

            if CounterfactualSetting.UNTARGETED_UNCATEGORIZED in settings:
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

            if CounterfactualSetting.TARGETED_UNCATEGORIZED in settings:
                for target_item in uncat_target_scores:
                    targ_uncat_log_ks = [
                        {
                            f"rankings@{k}": seq_tostr(uncat_target_gt[k]),
                            f"counter_rankings@{k}": seq_tostr(uncat_rankings[k]),
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

            if CounterfactualSetting.TARGETED_CATEGORIZED in settings:
                for target_cat in cat_target_scores:
                    targ_cat_log_ks = [
                        {
                            f"rankings@{k}": seq_tostr(cat_target_gt[k]),
                            f"counter_rankings@{k}": seq_tostr(cat_rankings[k]),
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
        untarg_logs = []
        targ_logs = []
        if CounterfactualSetting.UNTARGETED_CATEGORIZED in settings:
            best_cat_log = min(cat_logs, key=lambda x: x["score@1"])
            untarg_logs.append(best_cat_log)
        if CounterfactualSetting.UNTARGETED_UNCATEGORIZED in settings:
            best_uncat_log = min(uncat_logs, key=lambda x: x["score@1"])
            untarg_logs.append(best_uncat_log)
        if CounterfactualSetting.TARGETED_CATEGORIZED in settings:
            best_targ_cat_logs = {
                t: min(
                    [log for log in targ_cat_logs if log["target"] == t],
                    key=lambda x: x["score@1"],
                )
                for t in {log["target"] for log in targ_cat_logs}
            }
            targ_logs.append(best_targ_cat_logs)
        if CounterfactualSetting.TARGETED_UNCATEGORIZED in settings:
            best_targ_uncat_logs = {
                t: min(
                    [log for log in targ_uncat_logs if log["target"] == t],
                    key=lambda x: x["score@1"],
                )
                for t in {log["target"] for log in targ_uncat_logs}
            }
            targ_logs.append(best_targ_uncat_logs)
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
        verbose: bool = True,
    ):
        super().__init__(
            input_seq=input_seq,
            model=model,
            alphabet=alphabet,
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
        for _ in range(ConfigParams.POP_SIZE):
            executed, mutable, fixed = self.split.apply(x)
            random_idxs = random.sample(range(len(mutable)), k=self.num_edits)
            random_chars = random.sample(self.alphabet, k=self.num_edits)
            for idx, char in zip(random_idxs, random_chars):
                mutable[idx] = char
            new_seq = executed + mutable + fixed
            try:
                logits = self.model(torch.tensor(new_seq).unsqueeze(0))
            except:
                print(f"[ERROR DEBUG] with char: {char}")
                raise
            x_primes.append((new_seq, logits))
        return x_primes


class EducatedStategy(BaselineStrategy):
    def __init__(
        self,
        input_seq: Tensor,
        model: Callable,
        split: Split,
        alphabet: List[int],
        num_edits: int,
        target: int | str,
        categorized: bool,
        ks: Optional[List[int]] = None,
        verbose: bool = True,
    ):
        super().__init__(
            input_seq=input_seq,
            model=model,
            alphabet=alphabet,
            verbose=verbose,
            num_edits=num_edits,
            target_items=[target] if not categorized else [],  # type: ignore
            target_cats=[target] if categorized else [],  # type: ignore
            split=split,
            ks=ks,
        )
        self.target = (
            target
            if categorized
            else token2id(token=str(target), dataset=ConfigParams.DATASET)
        )
        self.categorized = categorized
        self.category_map = get_category_map()

    def generate(self) -> List[Tuple[List[int], Tensor]]:
        x = trim(self.input_seq.squeeze(0)).tolist()
        x_primes = []
        for _ in range(ConfigParams.POP_SIZE):
            executed, mutable, fixed = self.split.apply(x)
            random_idxs = random.sample(range(len(mutable)), k=self.num_edits)

            if self.categorized:
                same_cat_items = self._get_same_cat_items(cat=self.target)  # type: ignore
                educated_chars = random.sample(
                    population=same_cat_items, k=self.num_edits
                )
            else:
                if self.num_edits > 1:
                    raise ValueError(
                        "In the targeted-uncategorized case, only a num_edits of 1 is supported (since items must not repeat)"
                    )
                educated_chars = [self.target for _ in range(self.num_edits)]

            for idx, char in zip(random_idxs, educated_chars):
                mutable[idx] = char
            new_seq = executed + mutable + fixed
            try:
                logits = self.model(torch.tensor(new_seq).unsqueeze(0))
            except:
                print(f"[ERROR DEBUG] with char: {char}")
                raise
            x_primes.append((new_seq, logits))
        return x_primes

    def log(
        self,
        i: int,
        dataset,
    ):
        settings = {
            (
                CounterfactualSetting.TARGETED_UNCATEGORIZED
                if not self.categorized
                else CounterfactualSetting.TARGETED_CATEGORIZED
            )
        }
        super().log(i=i, dataset=dataset, baseline_name="educated", settings=settings)

    def _get_same_cat_items(self, cat: str):
        return [item for item in self.alphabet if cat in self.category_map[item]]


class PopularStrategy(GenerationStrategy):
    # TODO: skip this for now, implement it later.
    pass


dataset_target_items = {
    RecDataset.ML_100K: [50, 411, 630, 1305],
    RecDataset.STEAM: [35140, 292140, 582160, 271590],
    RecDataset.ML_1M: [2858, 2005, 728, 2738],
}

dataset_target_cats = {
    RecDataset.ML_100K: [
        "Horror",
        "Action",
        "Adventure",
        "Animation",
        "Fantasy",
        "Drama",
    ],
    RecDataset.STEAM: ["Action", "Indie", "Free to Play", "Sports", "Photo Editing"],
    RecDataset.ML_1M: [
        "Horror",
        "Action",
        "Adventure",
        "Animation",
        "Fantasy",
        "Drama",
    ],
}


def run_random_baseline(num_samples: Optional[int], num_edits: int):
    conf = get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)
    model = generate_model(config=conf)
    seqs = SequenceGenerator(conf)
    target_cats = dataset_target_cats[ConfigParams.DATASET]
    target_items = dataset_target_items[ConfigParams.DATASET]
    total = 0
    for i in seqs:
        total += 1
    print(f"[DEBUG] resetting")
    seqs.reset()
    sampled_indices = None
    if num_samples:
        sample_range = range(total)
        if num_samples > len(sample_range):
            raise ValueError(
                f"sample_num ({num_samples}) must be smaller than sample range ({len(sample_range)})"
            )
        sampled_indices = set(random.sample(population=sample_range, k=num_samples))
        assert len(sampled_indices) == num_samples
        total = num_samples
    pbar = tqdm(total=total, desc="Evaluating RandomStrategy baseline...")
    for i, seq in enumerate(seqs):
        if sampled_indices and i not in sampled_indices:
            continue
        strat = RandomStrategy(
            input_seq=seq,
            model=model,
            split=Split(None, 10, None),
            alphabet=list(get_items()),
            num_edits=num_edits,
            target_cats=target_cats,
            target_items=target_items,
        )

        exists = strat.logger.exists(
            log={"i": i, "baseline": "random", "num_edits": num_edits},
            primary_key=["i", "baseline", "num_edits"],
            consider_config=True,
        )
        if exists:
            pbar.total -= 1
            continue
        pop = strat.generate()
        try:
            strat.log(i, pop, baseline_name="random")
        except KeyError:
            pbar.update(1)
            continue
        pbar.update(1)


def run_educated_baseline(num_samples: Optional[int], num_edits: int):
    conf = get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)
    model = generate_model(config=conf)
    seqs = SequenceGenerator(conf)
    target_cats = dataset_target_cats[ConfigParams.DATASET]
    target_items = dataset_target_items[ConfigParams.DATASET]
    total = 0
    for i in seqs:
        total += 1
    sample_range = range(total)
    total *= len(target_cats) * len(target_items)
    seqs.reset()
    sampled_indices = None
    if num_samples:
        if num_samples > len(sample_range):
            raise ValueError(
                f"sample_num ({num_samples}) must be smaller than sample range ({len(sample_range)})"
            )
        sampled_indices = set(random.sample(population=sample_range, k=num_samples))
        assert len(sampled_indices) == num_samples
        total = num_samples
    pbar = tqdm(total=total, desc="Evaluating EducatedStategy baseline...")
    for categorized in [True, False] if num_edits == 1 else [True]:
        target_list = target_cats if categorized else target_items
        for target in target_list:
            for i, seq in enumerate(seqs):
                pbar.set_postfix_str(
                    f"{'categorized' if categorized else 'uncategorized'} | target: {target}"
                )
                if sampled_indices and i not in sampled_indices:
                    continue
                strat = EducatedStategy(
                    input_seq=seq,
                    model=model,
                    split=Split(None, 10, None),
                    alphabet=list(get_items()),
                    num_edits=num_edits,
                    categorized=categorized,
                    target=target,
                )
                exists = strat.logger.exists(
                    log={
                        "i": i,
                        "baseline": "educated",
                        "categorized": f"{categorized}",
                        "target": target,
                        "num_edits": num_edits,
                    },
                    primary_key=["i", "baseline", "categorized", "target", "num_edits"],
                    consider_config=True,
                )
                if exists:
                    pbar.total -= 1
                    continue
                pop = strat.generate()
                try:
                    strat.log(i, pop)
                except KeyError:
                    pbar.update(1)
                    continue
                pbar.update(1)
            seqs.reset()


def main(baseline: str, num_samples: int = 200, num_edits: int = 2):
    if baseline == "random":
        run_random_baseline(num_samples, num_edits)
    elif baseline == "educated":
        run_educated_baseline(num_samples, num_edits)
    else:
        raise ValueError(f"Baseline {baseline} not supported yet")


if __name__ == "__main__":
    fire.Fire(main)
