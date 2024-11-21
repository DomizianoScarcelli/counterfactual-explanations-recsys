import os
from typing import Generator, Optional

import fire
import pandas as pd
import torch
from pandas import DataFrame
from recbole.model.abstract_recommender import SequentialRecommender
from tqdm import tqdm

from alignment.actions import print_action
from config import DATASET, GENERATIONS, HALLOFFAME_RATIO, MODEL, POP_SIZE
from constants import MAX_LENGTH
from exceptions import (CounterfactualNotFound, DfaNotAccepting,
                        DfaNotRejecting, NoTargetStatesError)
from genetic.dataset.generate import dataset_generator, interaction_generator
from models.config_utils import generate_model, get_config
from models.utils import pad, trim
from performance_evaluation.alignment.utils import (evaluate_stats, get_split,
                                                    is_already_evaluated,
                                                    log_run,
                                                    preprocess_interaction)
from run import single_run, timed_learning_pipeline, timed_trace_disalignment
from utils import TimedGenerator, set_seed
from utils_classes.Split import Split


def evaluate_trace_disalignment(interactions: Generator, 
                                datasets: TimedGenerator, 
                                oracle: SequentialRecommender,
                                split_type: str,
                                use_cache: bool,
                                num_counterfactuals: int=100):
    good, bad, not_found, skipped = 0, 0, 0, 0
    log: DataFrame  = DataFrame({})
    genetic_params = (POP_SIZE, GENERATIONS, HALLOFFAME_RATIO)
    if os.path.exists("run.csv"):
        log = pd.read_csv("run.csv")

    error_messages = {
            DfaNotAccepting: "DfaNotAccepting",
            DfaNotRejecting: "DfaNotRejecting",
            NoTargetStatesError: "NoTargetStatesError",
            CounterfactualNotFound: "CounterfactualNotFound"
            }

    status = "unknown"
    for i, ((train, _), interaction) in enumerate(tqdm(zip(datasets, interactions), desc="Performance evaluation...")):
        if i == num_counterfactuals:
            print(f"Generated {num_counterfactuals}, exiting...")
            break
        source_sequence, source_gt = preprocess_interaction(interaction, oracle)
        print(f"Source sequence:", source_sequence)
        time_dataset_generation = datasets.get_times()[i]
        splits_key, splits = get_split(len(source_sequence), split_type)
        splits = Split(*splits)
        
        if is_already_evaluated(log=log, sequence=source_sequence, splits_key=split_type):
            print(f"Splits {splits} already evaluated for the current trace, skipping...")
            continue
        try:
            aligned, cost, alignment = single_run(source_sequence=source_sequence, _dataset=train, split=splits)
        except (DfaNotAccepting, DfaNotRejecting, NoTargetStatesError, CounterfactualNotFound) as e:
            print(e)
            skipped += 1
            log = log_run(log, 
                           original=source_sequence,
                           splits_key=splits_key,
                           genetic_key=genetic_params,
                           alignment=None,
                           status=error_messages[type(e)], cost=0,
                           time_dataset_generation=time_dataset_generation,
                           time_automata_learning=0,
                           time_alignment=0,
                           use_cache=use_cache)
            continue
        if len(aligned) == MAX_LENGTH:
            aligned = torch.tensor(aligned).unsqueeze(0).to(torch.int64)
        elif len(aligned) < MAX_LENGTH:
            aligned = pad(trim(torch.tensor(aligned)), MAX_LENGTH).unsqueeze(0).to(torch.int64)
        else:
            skipped += 1
            log = log_run(log, 
                           original=source_sequence,
                           alignment=[print_action(a) for a in alignment],
                           splits_key=splits_key,
                           genetic_key=genetic_params,
                           status="MaximumLengthReached", 
                           cost=cost,
                           time_dataset_generation=time_dataset_generation,
                           time_automata_learning=timed_learning_pipeline.get_last_time(),
                           time_alignment=timed_trace_disalignment.get_last_time(),
                           use_cache=use_cache)
            continue
        print(f"Alignment:", [print_action(a) for a in alignment])
        try:
            aligned_gt = oracle.full_sort_predict(aligned).argmax(-1).item()
        except IndexError as e:
            print("IndexError on sequence ", source_sequence)
            raise e
        correct = source_gt != aligned_gt
        if correct: 
            status = "good"
            good += 1
            print(f"Good counterfactual! {source_gt} != {aligned_gt}")
        else:
            status = "bad"
            bad += 1
            print(f"Bad counterfactual! {source_gt} == {aligned_gt}")
        print(f"[{i}] Good: {good}, Bad: {bad}, Not Found: {not_found}, Skipped: {skipped}")
        log = log_run(log, 
                       original=source_sequence,
                       alignment=[print_action(a) for a in alignment],
                       splits_key=splits_key,
                       genetic_key=genetic_params,
                       status=status, 
                       cost=cost,
                       time_dataset_generation=time_dataset_generation,
                       time_automata_learning=timed_learning_pipeline.get_last_time(),
                       time_alignment=timed_trace_disalignment.get_last_time(),
                       use_cache=use_cache)


def main(mode: str = "evaluate", 
         use_cache: bool = True, 
         evaluation_log: Optional[str] = None,
         stats_output: Optional[str] = None,
         split_type: str = "1_mut"):
    set_seed()
    if mode == "evaluate":
        config = get_config(dataset=DATASET, model=MODEL)
        oracle: SequentialRecommender = generate_model(config)
        interactions = interaction_generator(config)
        datasets = TimedGenerator(dataset_generator(config=config, use_cache=use_cache))
        evaluate_trace_disalignment(interactions=interactions,
                                    datasets=datasets, 
                                    oracle=oracle,
                                    split_type=split_type, 
                                    use_cache=use_cache)
    elif mode == "stats":
        if not evaluation_log:
            raise ValueError("Evaluation log path needed for stats")
        if not os.path.exists(evaluation_log):
            raise FileNotFoundError(f"File {evaluation_log} does not exists")
        if not stats_output:
            raise ValueError("Evaluation stats output needed")
        evaluate_stats(evaluation_log, stats_output)

    else:
        raise ValueError(f"Mode {mode} not supported, choose between [evaluate, stats]")



if __name__ == "__main__":
    fire.Fire(main)
    
