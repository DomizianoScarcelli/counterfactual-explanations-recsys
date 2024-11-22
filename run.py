import time
import warnings
from typing import List, Tuple, Optional

from exceptions import (CounterfactualNotFound, DfaNotAccepting,
                        DfaNotRejecting, NoTargetStatesError)
import fire

from alignment.alignment import trace_disalignment
from automata_learning.learning import learning_pipeline
from config import DATASET, MODEL
from utils_classes.generators import DatasetGenerator, InteractionGenerator
from models.config_utils import get_config, generate_model
from performance_evaluation.alignment.utils import preprocess_interaction
from type_hints import Dataset, RecDataset, RecModel, SplitTuple
from utils import TimedFunction
from utils_classes.Split import Split
from alignment.utils import postprocess_alignment

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

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
          end_i: {end_i} {f"(will be {start_i+1})" if end_i is None else ""}
          splits: {splits} {"(will be (0, None, 0))" if not splits else ""}

          ---model--------------
          dataset_type: {dataset_type}
          model_type: {model_type}

          ---genetic algorithm---
          num_generations: {num_generations}
          dataset_examples: {dataset_examples}
          -----------------------
          """)
    
    #TODO: dynamically defined num_generations and dataset examples are still not being used. They can only be changed in the config.toml file

    #Init config
    config = get_config(dataset=dataset_type, model=model_type)
    model = generate_model(config)
    datasets = DatasetGenerator(config=config, use_cache=use_cache, return_interaction=True)
    
    #Parse args
    if splits:
        splits: SplitTuple = tuple(splits) #type: ignore
        if isinstance(splits, tuple):
            splits: List[Split] = [Split(*splits)] #type: ignore
        elif isinstance(splits, list) and isinstance(splits[0], tuple):
            splits: List[Split] = [Split(*s) for s in splits] #type: ignore
    
        assert isinstance(splits, list) and isinstance(splits[0], Split), f"Malformed splits: {splits}"
    else:
        splits = [Split(0, None, 0)]

    if end_i is None:
        end_i = start_i + 1
    assert start_i < end_i, f"Start index must be strictly less than end index: {start_i} < {end_i}"
    
    #Running loop
    i=0
    while True:
        # Execute only the loops where start_i <= i < end_i
        if i < start_i:
            print(f"Skipping i = {i}")
            datasets.skip()
            i += 1
            continue
        if i >= end_i:
            print(f"Generated {end_i}, exiting...")
            break
        
        assert datasets.index == i
        dataset, interaction = next(datasets)

        start = time.time()
        try:
            source_sequence, source_gt = preprocess_interaction(interaction, model)
        except (DfaNotAccepting, DfaNotRejecting, NoTargetStatesError, CounterfactualNotFound) as e:
            print(f"Raised {type(e)}")
            i += 1
            continue
        
        for split in splits:
            split = split.parse_nan(source_sequence)
            print(f"----RUN DEBUG-----")
            print(f"Current Split: {split}")
            aligned, cost, _ = single_run(source_sequence, dataset, split)
            end = time.time()
            align_time = end-start
            print(f"[{i}] Align time: {align_time}")
            print(f"[{i}] Alignment cost: {cost}")

            aligned_gt = model(aligned).argmax(-1).item()

            if source_gt == aligned_gt:
                print(f"[{i}] Bad counterfactual! {source_gt} == {aligned_gt}")
            else:
                print(f"[{i}] Good counterfactual! {source_gt} != {aligned_gt}")
            print("--------------------")

        i += 1
        
if __name__ == '__main__':
  fire.Fire(main)
