from utils_classes.distances import ndcg
from typing import Callable
import json
import pickle
import random
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Dict, List, Literal, Set, Tuple, TypedDict, overload

import _pickle as cPickle
from recbole.data.dataset.sequential_dataset import SequentialDataset
from torch import Tensor

from config import ConfigParams
from constants import cat2id
from exceptions import EmptyDatasetError
from type_hints import CategorizedDataset, CategorySet, Dataset, RecDataset
from utils_classes.Cached import Cached
from utils_classes.distances import edit_distance, intersection_weighted_ndcg


class ItemInfo(TypedDict):
    name: str
    category: List[str]


class NumItems(Enum):
    ML_100K = 1682
    ML_1M = 3703
    MOCK = 6


def _compare_int_ys(y1: int, y2: int, return_score: bool = False):
    if return_score:
        return y1 == y2, int(y1 == y2)
    return y1 == y2


def _compare_ndcg_ys(
    y1: CategorySet | List[CategorySet],
    y2: CategorySet | List[CategorySet],
    score_fn: Callable,
    return_score: bool = False,
) -> bool | Tuple[bool, float]:
    if isinstance(y1, set):
        y1 = [y1]
    if isinstance(y2, set):
        y2 = [y2]

    score = score_fn(y1, y2)
    equal = score >= ConfigParams.THRESHOLD
    if return_score:
        return equal, score
    return equal


def equal_ys(
    gt: int | CategorySet | List[CategorySet] | Tensor,
    pred: int | CategorySet | List[CategorySet] | Tensor,
    return_score: bool = False,
):
    """
    Compares model outputs abstracting their type. It works with:
        - int and torch.Tensor, comparing them with the equal (==) operator
        - Set[int], comparing them with a thresholded jaccard similarity
    """
    if isinstance(gt, (int, Tensor)) and isinstance(pred, (int, Tensor)):
        if isinstance(gt, Tensor) and len(gt.flatten()) == 1:
            gt = gt.item()  # type: ignore

        if isinstance(pred, Tensor) and len(pred.flatten()) == 1:
            pred = pred.item()  # type: ignore

        if isinstance(gt, int) and isinstance(pred, int):
            print("[DEBUG] comparing with int comparison")
            return _compare_int_ys(gt, pred, return_score=return_score)  # type: ignore

    if isinstance(gt, (list, Tensor)) and isinstance(pred, (list, Tensor)):
        if isinstance(gt[0], Tensor):
            gt = [x.item() for x in gt]  # type: ignore
        if isinstance(pred[0], Tensor):
            pred = [x.item() for x in pred]  # type: ignore

        if all(isinstance(x, int) for x in (gt + pred)):
            print("[DEBUG] comparing with ndcg")
            return _compare_ndcg_ys(gt, pred, return_score=return_score, score_fn=ndcg)
    if (
        isinstance(gt, set)
        or isinstance(pred, set)
        or (
            isinstance(gt, list)
            and isinstance(gt[0], set)
            and isinstance(pred, list)
            and isinstance(pred[0], set)
        )
    ):
        print("[DEBUG] comparing with intersection_weighted_ndcg")
        return _compare_ndcg_ys(
            gt, pred, return_score=return_score, score_fn=intersection_weighted_ndcg
        )
    raise ValueError(f"Types {type(gt)} and {type(pred)} not supported")


def get_items(dataset: RecDataset = ConfigParams.DATASET) -> Set[int]:
    category_map = get_category_map(dataset)
    items = set(int(x) for x in category_map.keys())
    return items


def get_category_map(dataset: RecDataset = ConfigParams.DATASET) -> Dict[int, str]:

    def load_json(path: Path):
        if not path.exists():
            raise FileNotFoundError(
                "Category map has not been found, generate it with `python -m scripts.create_category_mapping`"
            )
        with open(path, "r") as f:
            data = json.load(f)

        return {int(key): value for key, value in data.items()}

    if dataset in [RecDataset.ML_1M, RecDataset.ML_100K]:
        category = Path(f"data/category_map_{dataset.value}.json")
    else:
        raise NotImplementedError(
            f"get_category_map not implemented for dataset {dataset}"
        )

    return Cached(category, load_fn=load_json).get_data()


@overload
def labels2cat(
    ys: List[int] | Tensor,
    encode: Literal[True],  # When encode is True
    dataset: RecDataset = ConfigParams.DATASET,
) -> List[CategorySet]:  # Encoded categories are sets of ints
    ...


@overload
def labels2cat(
    ys: List[int] | Tensor,
    encode: Literal[False],  # When encode is False
    dataset: RecDataset = ConfigParams.DATASET,
) -> List[Set[str]]:  # Unencoded categories are sets of strings
    ...


def labels2cat(
    ys: List[int] | Tensor,
    encode: bool = True,
    dataset: RecDataset = ConfigParams.DATASET,
) -> List[CategorySet] | List[Set[str]]:
    itemid2cat = get_category_map(dataset)
    if isinstance(ys, Tensor):
        ys = ys.tolist()

    if encode:
        return [set(cat2id[cat] for cat in itemid2cat[y]) for y in ys]  # type: ignore
    return [set(itemid2cat[y]) for y in ys]


def label2cat(
    label: int, dataset: RecDataset = ConfigParams.DATASET, encode: bool = True
) -> List[int]:
    itemid2cat = get_category_map(dataset)
    categories = itemid2cat[label]
    if not encode:
        return categories
    return [cat2id[cat] for cat in categories]


def get_remapped_dataset(dataset: RecDataset) -> SequentialDataset:
    def load_pickle(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    if dataset in [RecDataset.ML_1M, RecDataset.ML_100K]:
        path = Path(f"data/{ConfigParams.DATASET.value}-SequentialDataset.pth")
    else:
        raise NotImplementedError(
            f"get_category_map not implemented for dataset {dataset}"
        )

    return Cached(path, load_fn=load_pickle).get_data()


def id2token(dataset: RecDataset, id: int) -> int:
    """Maps interal item ids to external tokens"""
    remapped_dataset = get_remapped_dataset(dataset)
    return int(remapped_dataset.id2token("item_id", ids=id))


def token2id(dataset: RecDataset, token: str) -> int:
    """Maps external item tokens to internal ids."""
    remapped_dataset: SequentialDataset = get_remapped_dataset(dataset)
    return int(remapped_dataset.token2id("item_id", tokens=token))


def get_item_info(datset: RecDataset, id: int) -> ItemInfo:
    """Returns the information about a certain item in the dataset"""
    # TODO:implement
    info = {"name": "", "category": []}
    return info


def clone(x):
    return cPickle.loads(cPickle.dumps(x))


def random_points_with_offset(max_value: int, max_offset: int):
    if max_value <= 2:
        return (0, 0)
    i = random.randint(0, max_value)
    j = random.randint(max(0, i - max_offset), min(max_value, i + max_offset))
    # Sort i and j to ensure i <= j
    return tuple(sorted([i, j]))


def _evaluate_generation(
    input_seq: Tensor, dataset: Dataset, label: int
) -> Tuple[float, Tuple[float, float]]:
    # Evaluate label
    same_label = sum(1 for ex in dataset if ex[1] == label)
    # Evaluate example similarity
    distances_norm = []
    distances_nnorm = []
    for seq, _ in dataset:
        distances_norm.append(edit_distance(input_seq, seq))
        distances_nnorm.append(edit_distance(input_seq, seq, normalized=False))
    return (same_label / len(dataset)), (mean(distances_norm), mean(distances_nnorm))


def _evaluate_categorized_generation(
    input_seq: Tensor, dataset: CategorizedDataset, cats: List[CategorySet]
) -> Tuple[float, Tuple[float, float]]:
    # Evaluate label
    equal_cats = sum(1 for _, cat in dataset if equal_ys(cats, cat))
    # Evaluate example similarity
    distances_norm = []
    distances_nnorm = []
    for seq, _ in dataset:
        distances_norm.append(edit_distance(input_seq, seq))
        distances_nnorm.append(edit_distance(input_seq, seq, normalized=False))
    if len(dataset) == 0:
        # TODO: is it correct to raise this exception?
        raise EmptyDatasetError(
            "Generated dataset has length 0, change the dataset generation parameters to be more loose"
        )
    means = (mean(distances_norm), mean(distances_nnorm))
    return (equal_cats / len(dataset)), (means)
