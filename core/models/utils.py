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
    assert length >= len(seq), f"Seq must not be longer than {length}, seq is: {seq} with shape: {seq.shape}"
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

def pad_batch(seqs: List[List[int]] | List[Tensor], length: int) -> Tensor:
    """
    Given a list of N Tensors[int] or a list of N List[int], it returns a tensor of
    tensors padded with PADDING_CHAR of shape [N, length]
    """
    if isinstance(seqs, Tensor):
        return torch.stack([pad(s, length) for s in seqs])
    return torch.stack([pad(torch.tensor(s), length) for s in seqs])


def replace_padding(seq: Tensor, pad_char: int, new_pad_char: int) -> Tensor:
    """
    Replaces the padding of a Tensor from the pad_char to the new_pad_char.

    Args:
        seq: A tensor of shape [B, N] if batched or [N] if not batched.
        pad_char: an integer representing the old padding char.
        new_pad_char: an interger representing the new padding char.

    Returns:
        A new tensor with the same shape of `seq`, where the pad_char has been
        replaced with new_pad_char
    """
    assert new_pad_char not in seq, f"Pad {new_pad_char} is in seq: {seq}"
    return torch.where(seq == pad_char, torch.tensor(new_pad_char), seq)

def topk(logits: Tensor, k: int, dim: int, indices: bool=False) -> Tensor:
    """
    Returns the top-k most probable items if indices is True, the logits if
    False, giving the logits that represent the likelihood. 

    Args:
        logits: Input tensor.
        k: Number of top elements to select.
        dim: Dimension along which to perform the top-k operation.

    Returns:
        Tensor: The top-k values along the specified dimension.
    """
    # Use PyTorch's built-in topk function
    topk_values, topk_indices = torch.topk(logits, k, dim=dim, largest=True, sorted=True)
    return topk_indices if indices else topk_values
