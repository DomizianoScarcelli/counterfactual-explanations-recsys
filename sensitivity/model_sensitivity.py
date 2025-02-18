from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from numpy import isin
import pandas as pd
import torch
from recbole.model.abstract_recommender import SequentialRecommender
from torch import Tensor
from tqdm import tqdm

from config import ConfigParams
from constants import cat2id
from generation.utils import equal_ys, get_items, labels2cat
from models.config_utils import generate_model, get_config
from models.utils import topk, trim
from type_hints import CategorySet, RecDataset
from utils import seq_tostr
from utils_classes.generators import (
    InteractionGenerator,
    SequenceGenerator,
    SkippableGenerator,
)
from utils_classes.RunLogger import RunLogger


def generate_sequence_variants(sequence: Tensor, position: int, alphabet: Tensor):
    """
    Generate a matrix of possible sequences by modifying a single position in the input sequence.

    This function takes an input sequence, modifies a specific position using each element in the given alphabet,
    and passes both the original and modified sequences through the provided model. It returns the model's output
    for the original sequence and the modified sequences.

    Args:
        sequence (Tensor): The input sequence as a tensor of shape `(1, seq_length)`.
        model (SequentialRecommender): The model used to generate outputs for the sequences.
        position (int): The index of the position in the sequence to be modified.
        alphabet (Tensor): A tensor containing the values to be used for modification.

    Returns:
        Tuple[Tensor, Tensor]:
            - `out`: The model's output for the original sequence.
            - `out_primes`: The model's outputs for the modified sequences, with shape `(len(alphabet), output_dim)`.

    Notes:
        - The `trim` function is assumed to preprocess the sequence by trimming unwanted padding or values.
        - The function ensures that the `position` is within the bounds of the sequence.
        - The tensor `x_primes` is created by repeating the original sequence for each element in the alphabet
          and substituting the value at the specified `position` with the corresponding alphabet value.

    Example:
        Given `sequence = [[1, 2, 3]]`, `position = 1`, and `alphabet = [4, 5]`:
        - The modified sequences are `[[1, 4, 3], [1, 5, 3]]`.
        - The function returns the model's output for `[[1, 2, 3]]` and `[[1, 4, 3], [1, 5, 3]]`.
    """
    x = sequence
    assert x.dim() == 2 and x.size(0) == 1

    if x.size(1) <= position:
        return

    x_primes = x.repeat(len(alphabet), 1)
    assert x_primes.shape == torch.Size(
        [len(alphabet), x.size(1)]
    ), f"x shape uncorrect: {x_primes.shape} != {[len(alphabet), x.size(1)]}"
    positions = torch.tensor([position] * len(alphabet))

    x_primes[torch.arange(len(alphabet)), positions] = alphabet

    return x_primes


def model_sensitivity_universal(
    sequences: SkippableGenerator,
    model: SequentialRecommender,
    position: int,
    ks: List[int],
    y_target: Optional[int] = None,
    log_path: Optional[Path] = None,
    targeted: bool = False,
    categorized: bool = False,
):
    """Analyze the sensitivity of a sequential recommender model to changes in category predictions
    when input sequences are modified at a specific position. Optionally logs the results to a file.
    """
    score_log = {f"score@{k}": float for k in ks}
    log = {
        "i": int,
        "position": int,
        "pos_from_end": int,
        "alphabet_len": int,
        "sequence": str,
        "model": str,
        "dataset": str,
        "targeted": bool,
        "categorized": bool,
        **score_log,
    }

    if log_path:
        logger = RunLogger(db_path=log_path, schema=log, add_config=False)

    if ConfigParams.DATASET in [RecDataset.ML_1M, RecDataset.ML_100K]:
        alphabet = torch.tensor(list(get_items()))
    else:
        raise NotImplementedError(f"Dataset {ConfigParams.DATASET} not supported yet!")

    if targeted and not y_target:
        raise ValueError("If setting is 'targeted', then y_target has to be defined")

    if targeted:
        y_targets_ks = {
            k: [{y_target} if categorized else y_target for _ in range(k)] for k in ks
        }

    i = 0
    start_i = 0
    end_i = sum(1 for _ in InteractionGenerator())  # all users
    pbar = tqdm(
        total=end_i - start_i,
        desc=f"Testing model sensitivity on position {position}",
        leave=False,
    )
    i_list, sequence_list = [], []
    scores_ks = {k: [] for k in ks}
    primary_key = ["i", "position"]
    if targeted:
        primary_key.append("target")
    for i, sequence in enumerate(sequences):
        if i < start_i:
            continue
        if i >= end_i:
            break

        pbar.update(1)
        if log_path:
            new_row = {"i": i, "position": position}
            if targeted:
                new_row["target"] = y_target

            if logger.exists(new_row, primary_key, consider_config=False):
                continue

        sequence = trim(sequence.squeeze(0)).unsqueeze(0)
        x_primes = generate_sequence_variants(sequence, position, alphabet)
        if x_primes is None:
            continue

        out = model(sequence)
        out_primes = model(x_primes)

        topk_out_ks = {
            k: topk(logits=out, k=k, dim=-1, indices=True).squeeze(0) for k in ks
        }
        out_cat_ks = topk_out_ks
        if categorized:
            out_cat_ks = {k: labels2cat(topk_out_ks[k]) for k in ks}  # type: ignore

        topk_out_primes_ks = {
            k: topk(logits=out_primes, k=k, dim=-1, indices=True) for k in ks
        }

        out_primes_cat_ks: Dict[int, Tensor] | Dict[int, List[List[CategorySet]]] = (
            topk_out_primes_ks
        )
        if categorized:
            out_primes_cat_ks = {
                k: [
                    labels2cat(topk_out_prime)  # type: ignore
                    for topk_out_prime in topk_out_primes_ks[k]
                ]
                for k in ks
            }

        # TODO: This can be vectorized by using torch operations
        scores_batch_ks = {k: [] for k in ks}
        for k in ks:
            n_items = len(out_primes_cat_ks[k])
            y = out_cat_ks[k]
            if targeted:
                y_targets = y_targets_ks[k]
            for n_i in range(n_items):
                y_prime = out_primes_cat_ks[k][n_i]

                assert len(y) == len(y_prime) == k

                if targeted:
                    eq_res = equal_ys(y_targets, y_prime, return_score=True)
                else:
                    eq_res = equal_ys(y, y_prime, return_score=True)
                assert isinstance(eq_res, tuple)
                _, score = eq_res
                scores_batch_ks[k].append(score)

            scores_ks[k].append(mean(scores_batch_ks[k]))
        i_list.append(i)
        sequence_list.append(sequence.squeeze().tolist())

    score_dict = {f"score@{k}": scores_ks[k] for k in ks}
    data = {
        "i": i_list,
        "position": [position] * len(i_list),
        "pos_from_end": [len(x) - (position + 1) for x in sequence_list],
        "alphabet_len": [len(alphabet)] * len(i_list),
        "sequence": [seq_tostr(x) for x in sequence_list],
        "model": [ConfigParams.MODEL.value] * len(i_list),
        "dataset": [ConfigParams.DATASET.value] * len(i_list),
        "targeted": [targeted] * len(i_list),
        "categorized": [categorized] * len(i_list),
        **score_dict,
    }
    if targeted:
        data["target"] = [y_target] * len(i_list)
    if log_path and len(i_list) > 0:
        for row_i in range(len(sequence_list)):
            row = {key: data[key][row_i] for key in data}
            logger.log_run(log=row, primary_key=primary_key, strict=True)
    else:
        print(pd.DataFrame(data))


def run_on_all_positions(
    ks: List[int],
    log_path: Optional[Path] = None,
):
    config = get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)
    sequences = SequenceGenerator(config)
    model = generate_model(config)
    start_i, end_i = 49, 0
    for i in tqdm(
        range(start_i, end_i - 1, -1),
        "Testing model sensitivity on all positions",
        leave=False,
    ):
        # This is very important in order to evaluate different positions on the same sequences
        sequences.reset()

        targeted = ConfigParams.TARGETED
        categorized = ConfigParams.CATEGORIZED

        y_target: Optional[int | str] = ConfigParams.TARGET_CAT if targeted else None

        if targeted:
            if y_target == False:
                raise ValueError(
                    "false -> run on all targets is not implemented in model sensitivity, please specify the target as a string (category) or int (item id)"
                )
            if not (
                isinstance(y_target, str)
                and categorized
                or isinstance(y_target, int)
                and not categorized
            ):
                raise ValueError(
                    f"if categorized, y_target must be str; if non categorized, y_target must be int. Now categorized={categorized} (type {type(categorized)}) and y_target is of type: {type(y_target)}"
                )

            if isinstance(y_target, str):
                y_target = cat2id[y_target]

        model_sensitivity_universal(
            model=model,
            sequences=sequences,
            position=i,
            log_path=log_path,
            ks=ks,
            y_target=y_target,
            targeted=targeted,
            categorized=categorized,
        )
