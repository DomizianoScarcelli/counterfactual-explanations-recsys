import time
import warnings
from typing import List, Tuple, Optional

import fire

from alignment.alignment import trace_disalignment
from automata_learning.learning import learning_pipeline
from config import DATASET, MODEL
from genetic.dataset.generate import dataset_generator, interaction_generator
from models.config_utils import get_config, generate_model
from performance_evaluation.alignment.utils import preprocess_interaction
from type_hints import Dataset, RecDataset, RecModel, SplitTuple
from utils import TimedFunction
from utils_classes.Split import Split
from alignment.utils import postprocess_alignment

warnings.simplefilter(action='ignore', category=FutureWarning)

timed_learning_pipeline = TimedFunction(learning_pipeline)
timed_trace_disalignment = TimedFunction(trace_disalignment)

def single_run(source_sequence: List[int], 
               _dataset: Tuple[Dataset, Dataset],
               split:Optional[Split]=None):
    assert isinstance(source_sequence, list), f"Source sequence is not a list, but a {type(source_sequence)}"
    assert isinstance(source_sequence[0], int), f"Elements of the source sequences are not ints, but {type(source_sequence[0])}"

    dfa = timed_learning_pipeline(source=source_sequence, dataset=_dataset)
    
    if split:
        source_sequence = split.apply(source_sequence) #type: ignore

    aligned, cost, alignment = timed_trace_disalignment(dfa, source_sequence)
    aligned = postprocess_alignment(aligned)
    return aligned, cost, alignment

def main(dataset_type:RecDataset=DATASET,
         model_type:RecModel=MODEL,
         start_i: int = 0,
         end_i: Optional[int]=None,
         splits: Optional[List[int]] = None, #type: ignore
         use_cache: bool=True,
         num_generations: int=20,
         dataset_examples: int=2000):
    print(f"""
          -----------------------
          parameters
          -----------------------
          use_cache: {use_cache}

          start_i: {start_i}
          end_i: {end_i}
          splits: {splits}

          ---model--------------
          dataset_type: {dataset_type}
          model_type: {model_type}

          ---genetic algorithm---
          num_generations: {num_generations}
          dataset_examples: {dataset_examples}
          -----------------------
          """)

    #Init config
    config = get_config(dataset=dataset_type, model=model_type)
    model = generate_model(config)
    interactions = interaction_generator(config)
    datasets = dataset_generator(config=config, use_cache=use_cache)
    
    #Parse args
    if splits:
        splits: SplitTuple = tuple(splits) #type: ignore
        if isinstance(splits, tuple):
            splits: List[Split] = [Split(*splits)] #type: ignore
        elif isinstance(splits, list) and isinstance(splits[0], tuple):
            splits: List[Split] = [Split(*s) for s in splits] #type: ignore
    
        assert isinstance(splits, list) and isinstance(splits[0], Split), f"Malformed splits: {splits}"

    if end_i is None:
        end_i = start_i + 1
    assert start_i < end_i, f"Start index must be strictly less than end index: {start_i} < {end_i}"
    
    #Running loop
    i=0
    while True:
        # Execute only the loops where start_i <= i < end_i
        if i < start_i:
            i += 1
            continue
        if i >= end_i:
            print(f"Generated {end_i}, exiting...")
            break
        dataset = next(datasets)
        interaction = next(interactions)

        start = time.time()
        source_sequence, source_gt = preprocess_interaction(interaction, model)

        aligned, cost, _ = single_run(source_sequence, dataset)
        end = time.time()
        align_time = end-start
        print(f"[{i}] Align time: {align_time}")
        print(f"[{i}] Alignment cost: {cost}")

        aligned_gt = model(aligned).argmax(-1).item()

        if source_gt == aligned_gt:
            print(f"[{i}] Bad counterfactual! {source_gt} == {aligned_gt}")
        else:
            print(f"[{i}] Good counterfactual! {source_gt} != {aligned_gt}")

        i += 1
        
if __name__ == '__main__':
  fire.Fire(main)
