from typing import List

import torch
from torch import Tensor


def trim(seq: Tensor | List[int]) -> Tensor:
    """ Trims the final -1s of the tensor

    Args:
        seq: tensor representing the sequence, padded with -1s

    Returns:
        the sequence without padded -1s
    """
    if isinstance(seq, Tensor):
        return _trim_tensor(seq)
    else:
        return torch.tensor(_trim_list(seq))


def _trim_tensor(seq: Tensor) -> Tensor:
    """ Trims the final -1s of the tensor

    Args:
        seq: tensor representing the sequence, padded with -1s

    Returns:
        the sequence without padded -1s
    """
    assert len(seq.shape) == 1, f"Sequence must have a single dim, {seq.shape}"
    positive_count = (seq >= 0).sum()
    n_seq, to_be_removed = seq[:positive_count], seq[positive_count:]
    assert len(n_seq) + len(to_be_removed) == len(seq), f"{len(n_seq)} + {len(to_be_removed)} != {len(seq)}, {n_seq}, {to_be_removed}"
    assert (n_seq >= 0).sum() == len(n_seq) and (to_be_removed >= 0).sum() == 0, f"""
    Sequence must use the character -1 only for padding!
    positive_count: {positive_count}
    n_seq: {n_seq}
    to_be_removed: {to_be_removed}
    """
    return n_seq


def _trim_list(seq: List[int]) -> List[int]:
    """ Trims the final -1s of the tensor

    Args:
        seq: tensor representing the sequence, padded with -1s

    Returns:
        the sequence without padded -1s
    """
    assert isinstance(seq[0], int)
    positive_count = sum(1 for c in seq if c >= 0)
    n_seq, to_be_removed = seq[:positive_count], seq[positive_count:]
    assert len(n_seq) + len(to_be_removed) == len(seq), f"{len(n_seq)} + {len(to_be_removed)} != {len(seq)}, {n_seq}, {to_be_removed}"
    assert sum(1 for c in n_seq if c >= 0) == len(n_seq) and sum(1 for c in to_be_removed if c >= 0) == 0, f"""
    Sequence must use the character -1 only for padding!
    positive_count: {positive_count}
    n_seq: {n_seq}
    to_be_removed: {to_be_removed}
    """
    return n_seq

def _pad_tensor(seq: Tensor, length: int) -> Tensor:
    """ Pads the sequence with -1 until it's length is equal to `length`

    Args:
        seq: tensor representing the sequence, padded with -1s
        length: the desired final length

    Returns:
        the sequence with padded -1s
    """
    assert len(seq.shape) == 1, f"Sequence must have a single dim, {seq.shape}"
    if len(seq) == length: return seq
    if (seq >= 0).sum() != len(seq):
        seq = trim(seq)
    return torch.cat((seq, torch.full(fill_value=-1, size=(length - len(seq),))))

def _pad_list(seq: List[int], length: int) -> List[int]:
    """ Pads the sequence with -1 until it's length is equal to `length`

    Args:
        seq: tensor representing the sequence, padded with -1s
        length: the desired final length

    Returns:
        the sequence with padded -1s
    """
    assert isinstance(seq[0], int)
    if len(seq) == length: return seq
    if sum(1 for c in seq if c >= 0) != len(seq):
        seq = trim(seq)
    return seq + [-1] * (length - len(seq))



def pad(seq: Tensor | List[int], length: int) -> Tensor:
    """ Pads the sequence with -1 until it's length is equal to `length`

    Args:
        seq: tensor representing the sequence, padded with -1s
        length: the desired final length

    Returns:
        the sequence with padded -1s
    """
    if isinstance(seq, Tensor):
        return _pad_tensor(seq, length)
    else:
        return torch.tensor(_pad_list(seq, length))

def pad_batch(seqs: List[Tensor], length: int) -> Tensor:
    return torch.stack([pad(torch.tensor(s), length) for s in seqs])
