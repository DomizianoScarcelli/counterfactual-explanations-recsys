from typing import Union

import torch
from recbole.model.sequential_recommender import BERT4Rec
from recbole.trainer import Interaction
from torch import Tensor

from utils import set_seed


class ExtendedBERT4Rec(BERT4Rec):
    def __init__(self, config, dataset):
        set_seed()
        super().__init__(config=config, dataset=dataset)
        self.eval()
        
        # self.item_length = torch.tensor([50])

    def full_sort_predict(self, interaction: Union[Interaction, Tensor]):
        if isinstance(interaction, Interaction):
            return super().full_sort_predict(interaction)
        elif isinstance(interaction, Tensor):
            return self._full_sort_predict_from_sequence(interaction)
        else:
            raise ValueError(f"Unsupported input type: {type(interaction)}")

    def _full_sort_predict_from_sequence(self, item_seq: Tensor):
        item_seq_len = torch.count_nonzero(item_seq).unsqueeze(0).to(torch.int64)
        item_seq = item_seq.to(torch.int64)
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B H]
        test_items_emb = self.item_embedding.weight[
            : self.n_items
        ]  # delete masked token
        scores = (
            torch.matmul(seq_output, test_items_emb.transpose(0, 1)) + self.output_bias
        )  # [B, item_num]
        return scores


    def batched_full_sort_predict(self, item_seq: Tensor):
        item_seq_len = torch.count_nonzero(item_seq, dim=1).to(torch.int64)
        item_seq = item_seq.to(torch.int64)
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B H]
        test_items_emb = self.item_embedding.weight[
            : self.n_items
        ]  # delete masked token
        scores = (
            torch.matmul(seq_output, test_items_emb.transpose(0, 1)) + self.output_bias
        )  # [B, item_num]
        return scores
