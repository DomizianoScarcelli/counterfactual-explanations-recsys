from typing import Union

import torch
from recbole.model.sequential_recommender import BERT4Rec
from recbole.trainer import Interaction
from torch import Tensor

from constants import PADDING_CHAR
from generation.dataset.utils import interaction_to_tensor
from models.utils import replace_padding


class ExtendedBERT4Rec(BERT4Rec):
    def __init__(self, config, dataset):
        super().__init__(config=config, dataset=dataset)
        self.eval()


    def __call__(self, x: Union[Interaction, Tensor]):
        return self.full_sort_predict(x)
        
    def full_sort_predict(self, interaction: Union[Interaction, Tensor]):
        if isinstance(interaction, Interaction):
            # sequence = get_sequence_from_interaction(interaction)
            return self.full_sort_predict_from_interaction(interaction)
        elif isinstance(interaction, Tensor):
            sequence = interaction
        else:
            raise ValueError(f"Unsupported input type: {type(interaction)}")
        return self.full_sort_predict_from_sequence(sequence)


    def full_sort_predict_from_interaction(self, interaction: Interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # print(f"DEBUG: item_seq: {item_seq}, \n item_seq_len: {item_seq_len}")
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        # print(f"DEBUG: Reconstructed item_seq: {item_seq}")
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B H] #type: ignore
        test_items_emb = self.item_embedding.weight[
            : self.n_items
        ]  # delete masked token
        scores = (
            torch.matmul(seq_output, test_items_emb.transpose(0, 1)) + self.output_bias
        )  # [B, item_num]
        return scores

    def full_sort_predict_from_sequence(self, item_seq: Tensor):
        with torch.no_grad():
            item_seq_len = (item_seq != PADDING_CHAR).sum(-1).to(torch.int64)
            item_seq = replace_padding(item_seq, PADDING_CHAR, 0).to(torch.int64)


            # print(f"DEBUG: item_seq: {item_seq}, \n item_seq_len: {item_seq_len} with shapes: {item_seq.shape}, {item_seq_len.shape}")

            # Shifts sequence by 1 and adds a mask token (NumItems+1) to the
            # last valid position (before padding)
            item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
            # print(f"DEBUG: Reconstructed item_seq: {item_seq}")
        
            seq_output = self.forward(item_seq) #[Batch_size B, max_length M, hidden_size H] (example torch.Size([1, 50, 64]))

            # Retrieves the embedding corresponding to the position where the
            # mask token was inserted
            seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B H]
            
            #Extracts the embedding for all the items (but not the one of the masked token).
            test_items_emb = self.item_embedding.weight[:self.n_items]  # [n_items, H], (example torch.Size([3708, 64]))
            
            # Dot product between the masks embeddings and each item's
            # embedding. It returns a [B, num_items] matrix of similarities
            # between the sequence embedding and all possible items.
            scores = (
                torch.matmul(seq_output, test_items_emb.transpose(0, 1)) + self.output_bias
            )  # [B, item_num]

            #The scores are the likelihood of each item being the next item in the sequence.
            return scores


