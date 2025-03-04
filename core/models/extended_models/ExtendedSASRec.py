from typing import Union

import torch
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Interaction
from torch import Tensor

from config.config import ConfigParams
from config.constants import PADDING_CHAR
from core.models.utils import replace_padding


class ExtendedSASRec(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config=config, dataset=dataset)
        self.eval()

    def __call__(self, x: Union[Interaction, Tensor]):
        x = x.to(ConfigParams.DEVICE)
        return self.full_sort_predict(x)
        
    def full_sort_predict(self, interaction: Union[Interaction, Tensor]):
        if isinstance(interaction, Interaction):
            return super().full_sort_predict(interaction)
        elif isinstance(interaction, Tensor):
            return self.full_sort_predict_from_sequence(interaction)
        else:
            raise ValueError(f"Unsupported input type: {type(interaction)}")

    def full_sort_predict_from_sequence(self, item_seq: Tensor):
        with torch.no_grad():
            item_seq_len = (item_seq != PADDING_CHAR).sum(-1).to(torch.int64)
            item_seq = replace_padding(item_seq, PADDING_CHAR, 0).to(torch.int64)
            seq_output = self.forward(item_seq, item_seq_len)
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
            return scores

