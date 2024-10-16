import torch
from torch import Tensor


def trim_zero(seq: Tensor) -> Tensor:
    """ Trims the final zeros of the tensor

    Args:
        seq: tensor representing the sequence with padded zeros

    Returns:
        the sequence without padded zeros
    """
    assert len(seq.shape) == 1, f"Sequence must have a single dim, {seq.shape}"
    nnz_count = seq.count_nonzero()
    n_seq, to_be_removed = seq[:nnz_count], seq[nnz_count:]
    assert n_seq.count_nonzero() == len(n_seq), "Sequence must use the character 0 only for padding!"
    assert len(to_be_removed) == len(seq) - nnz_count, f"Something went wrong! {seq} -> {n_seq}, {to_be_removed}"
    return n_seq


def pad_zero(seq: Tensor, length: int) -> Tensor:
    """ Pads the sequence with zeros until it's length is equal to `length`

    Args:
        seq: tensor representing the sequence with padded zeros
        length: the desired final length

    Returns:
        the sequence with padded zeros
    """
    assert len(seq.shape) == 1, f"Sequence must have a single dim, {seq.shape}"
    if len(seq) == length: return seq
    assert seq.count_nonzero() == len(seq), f"Sequence must not contain the character 0!: {seq}"
    return torch.cat((seq, torch.zeros((length - len(seq),))))
