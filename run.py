import time
import warnings
from typing import List, Tuple

import fire

from automata_learning import learning_pipeline
from config import DATASET, MODEL
from recommenders.generate_dataset import (dataset_generator, get_config,
                                           get_sequence_from_interaction,
                                           interaction_generator)
from recommenders.utils import trim_zero
from trace_alignment import trace_disalignment
from type_hints import Dataset, RecDataset, RecModel
from utils import TimedFunction

warnings.simplefilter(action='ignore', category=FutureWarning)

timed_learning_pipeline = TimedFunction(learning_pipeline)
timed_trace_disalignment = TimedFunction(trace_disalignment)

def single_run(source_sequence: List[int], _dataset: Tuple[Dataset, Dataset]):
    assert isinstance(source_sequence, list), f"Source sequence is not a list, but a {type(source_sequence)}"
    assert isinstance(source_sequence[0], int), f"Elements of the source sequences are not ints, but {type(source_sequence[0])}"

    dfa = timed_learning_pipeline(source=source_sequence, dataset=_dataset)
    aligned, cost, alignment = timed_trace_disalignment(dfa, source_sequence)
    return aligned, cost, alignment

def main(dataset:RecDataset=DATASET, 
         model:RecModel=MODEL, 
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
