from automata_utils import run_automata
from recommenders.utils import trim_zero
from trace_alignment import trace_alignment, trace_disalignment
from automata_learning import learning_pipeline
from recommenders.generate_dataset import (dataset_generator, get_config,
                                           interaction_generator,
                                           get_sequence_from_interaction)
from type_hints import (Dataset, RecModel, RecDataset, LabeledTensor)
import fire
from torch import Tensor
import time
from typing import Tuple, List
from recbole.trainer import Interaction

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def single_run(source_sequence: List[int], _dataset: Tuple[Dataset, Dataset]):
    assert isinstance(source_sequence, list), f"Source sequence is not a list, but a {type(source_sequence)}"
    assert isinstance(source_sequence[0], int), f"Elements of the source sequences are not ints, but {type(source_sequence[0])}"

    dfa = learning_pipeline(source=source_sequence, dataset=_dataset)
    aligned, cost, alignment = trace_disalignment(dfa, source_sequence)
    print(f"ALIGNED! {source_sequence} -> {aligned}, C={cost}")
    return aligned, cost, alignment

def main(dataset:RecDataset=RecDataset.ML_1M, 
         model:RecModel=RecModel.BERT4Rec, 
         num_counterfactuals: int=1,
         num_generations: int=20,
         dataset_examples: int=2000):
    print(f"""
          -----------------------
          PARAMETERS
          -----------------------

          ---Model--------------
          dataset: {dataset}
          model: {model}
          num_counterfactuals: {num_counterfactuals}

          ---Genetic Algorithm---
          num_generations: {num_generations}
          dataset_examples: {num_generations}
          -----------------------
          """)
    config = get_config(dataset=dataset, model=model)
    interactions = interaction_generator(config)
    datasets = dataset_generator(config=config)
    for i, (_dataset, interaction) in enumerate(zip(datasets, interactions)):

        start = time.time()
        if i == num_counterfactuals:
            print(f"Generated {num_counterfactuals}, exiting...")
            break
        source_sequence = get_sequence_from_interaction(interaction).squeeze(0)
        source_sequence = trim_zero(source_sequence)
        aligned, cost, _ = single_run(source_sequence.tolist(), _dataset)
        end = time.time()
        align_time = end-start
        print(f"[{i}] Align time: {align_time}")
        
if __name__ == '__main__':
  fire.Fire(main)
