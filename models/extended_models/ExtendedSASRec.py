from typing import Union

import torch
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Interaction
from torch import Tensor

from utils import set_seed


class ExtendedSASRec(SASRec):
    def __init__(self, config, dataset):
        set_seed()
        super().__init__(config=config, dataset=dataset)
        self.eval()

    def full_sort_predict(self, interaction: Union[Interaction, Tensor]):
        if isinstance(interaction, Interaction):
            return super().full_sort_predict(interaction)
        elif isinstance(interaction, Tensor):
            return self.full_sort_predict_from_sequence(interaction)
        else:
            raise ValueError(f"Unsupported input type: {type(interaction)}")

    def full_sort_predict_from_sequence(self, item_seq: Tensor):
        item_seq_len = torch.count_nonzero(item_seq, dim=-1).unsqueeze(0).to(torch.int64)
        item_seq = item_seq.to(torch.int64)
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

