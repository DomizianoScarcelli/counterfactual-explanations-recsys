from aalpy.automata.Dfa import Dfa, DfaState
from automata_utils import invert_automata
from dataset_generator import NumItems
from graph_search import (decode_action, get_shortest_alignment_dijkstra, 
                          get_shortest_alignment_a_star)
from tqdm import tqdm
from graph_search import Action, decode_action, act_str
from torch import Tensor
from recbole.config import Config
from models.ExtendedBERT4Rec import ExtendedBERT4Rec
from recommenders.generate_dataset import generate_model
from run import single_run
from type_hints import (Dataset, RecModel, RecDataset, LabeledTensor)
from recommenders.generate_dataset import (dataset_generator, get_config,
                                           interaction_generator,
                                           get_sequence_from_interaction)
import torch


def evaluate_trace_disalignment(interactions, 
                                datasets, 
                                oracle,
                                num_counterfactuals: int=20):
    good, bad = 0, 0
    for i, (_dataset, interaction) in enumerate(tqdm(zip(datasets, interactions), desc="Performance evaluation...")):
        if i == num_counterfactuals:
            print(f"Generated {num_counterfactuals}, exiting...")
            break
        source_sequence = get_sequence_from_interaction(interaction)
        print(f"Source sequence:", source_sequence.tolist())
        source_gt = oracle.full_sort_predict(source_sequence).argmax(-1).item()
        aligned, cost = single_run(source_sequence, _dataset)
        # aligned_size = len(aligned)
        # aligned = torch.cat( (torch.tensor(aligned), torch.zeros((aligned_size,)) )).unsqueeze(0)
        #TODO: does it need zero padding?
        aligned = torch.tensor(aligned).unsqueeze(0).to(torch.int64)
        print(f"Aligned sequence:", aligned.tolist())
        aligned_gt = oracle.full_sort_predict(aligned).argmax(-1).item()
        correct = source_gt != aligned_gt
        # assert correct, "Source and aligned have the same label {source_gt} == {aligned_gt}"
        if correct: 
            good += 1
            print(f"Good counterfactual! {source_gt} != {aligned_gt}")
        else:
            bad += 1
            print(f"Bad counterfactual! {source_gt} == {aligned_gt}")
        print(f"[{i}] Good: {good}, Bad: {bad}")


if __name__ == "__main__":
    config = get_config(dataset=RecDataset.ML_1M, model=RecModel.BERT4Rec)
    oracle: ExtendedBERT4Rec = generate_model(config)
    interactions = interaction_generator(config)
    datasets = dataset_generator(config=config)
    evaluate_trace_disalignment(interactions, datasets, oracle)
