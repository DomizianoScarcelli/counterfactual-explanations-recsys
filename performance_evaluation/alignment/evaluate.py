import json
import os
from typing import Dict, List, Optional

import fire
import torch
from recbole.model.abstract_recommender import SequentialRecommender
from tqdm import tqdm

from alignment.actions import print_action
from config import DATASET, MODEL
from constants import MAX_LENGTH
from exceptions import (CounterfactualNotFound, DfaNotAccepting,
                        DfaNotRejecting, NoTargetStatesError)
from genetic.dataset.generate import dataset_generator, interaction_generator
from genetic.dataset.utils import get_sequence_from_interaction
from models.config_utils import generate_model, get_config
from models.utils import pad, trim
from performance_evaluation.alignment.utils import evaluate_stats, save_log
from run import single_run, timed_learning_pipeline, timed_trace_disalignment
from type_hints import Split, Trace
from utils import TimedGenerator, set_seed


def evaluate_trace_disalignment(interactions: TimedGenerator, 
                                datasets: TimedGenerator, 
                                oracle: SequentialRecommender,
                                num_counterfactuals: int=20,
                                force: bool=False):
    good, bad, not_found, skipped = 0, 0, 0, 0
    evaluation_log: Dict[str, List[Dict]] = {}
    if os.path.exists("evaluation_log.json"):
        with open("evaluation_log.json", "r") as f:
            evaluation_log = json.load(f)

    status = "unknown"
    for i, ((train, _), interaction) in enumerate(tqdm(zip(datasets, interactions), desc="Performance evaluation...")):
        if i == num_counterfactuals:
            print(f"Generated {num_counterfactuals}, exiting...")
            break
        source_sequence = get_sequence_from_interaction(interaction)
        source_gt = oracle.full_sort_predict(source_sequence).argmax(-1).item()
        source_sequence = trim(source_sequence.squeeze(0)).tolist()
        print(f"Source sequence:", source_sequence)
        time_dataset_generation = datasets.get_times()[i]
        splits: Split = (35/50, 15/50, 0/50)
        # splits: Split = (0, 1, 0)
        splits_key = ", ".join(f"{i}" for i in splits)

        if splits_key in evaluation_log and source_sequence in [run["original"] for run in evaluation_log[splits_key]] and not force:
            raise ValueError(f"Splits {splits} already evaluated for the current trace, set force=True if you want to override them")

        evaluation_log[splits_key] = []
        try:
            aligned, cost, alignment = single_run(source_sequence=source_sequence, _dataset=train, splits=splits)
        except (DfaNotAccepting, DfaNotRejecting, NoTargetStatesError, CounterfactualNotFound) as e:
            print(e)
            error_messages = {
                    DfaNotAccepting: "DfaNotAccepting",
                    DfaNotRejecting: "DfaNotRejecting",
                    NoTargetStatesError: "NoTargetStatesError",
                    CounterfactualNotFound: "CounterfactualNotFound"
                    }
            skipped += 1
            evaluation_log = save_log(evaluation_log, 
                                      original=source_sequence,
                                      splits_key=splits_key,
                                      alignment=None,
                                      status=error_messages[type(e)], cost=0,
                                      time_dataset_generation=time_dataset_generation,
                                      time_automata_learning=0,
                                      time_alignment=0)
            continue

        if len(aligned) == MAX_LENGTH:
            aligned = torch.tensor(aligned).unsqueeze(0).to(torch.int64)
        elif len(aligned) < MAX_LENGTH:
            aligned = pad(trim(torch.tensor(aligned)), MAX_LENGTH).unsqueeze(0).to(torch.int64)
        else:
            skipped += 1
            evaluation_log = save_log(evaluation_log, original=source_sequence,
                                      alignment=[print_action(a) for a in alignment],
                                      splits_key=splits_key,
                                      status="MaximumLengthReached", 
                                      cost=cost,
                                      time_dataset_generation=time_dataset_generation,
                                      time_automata_learning=timed_learning_pipeline.get_last_time(),
                                      time_alignment=timed_trace_disalignment.get_last_time())
            continue
        print(f"Alignment:", [print_action(a) for a in alignment])
        aligned_gt = oracle.full_sort_predict(aligned).argmax(-1).item()
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
        evaluation_log = save_log(evaluation_log, original=source_sequence,
                                  alignment=[print_action(a) for a in alignment],
                                  splits_key=splits_key,
                                  status=status, 
                                  cost=cost,
                                  time_dataset_generation=time_dataset_generation,
                                  time_automata_learning=timed_learning_pipeline.get_last_time(),
                                  time_alignment=timed_trace_disalignment.get_last_time())


def main(mode: str = "evaluate", 
         use_cache: bool = True, 
         evaluation_log: Optional[str] = None,
         stats_output: Optional[str] = None):
    set_seed()
    if mode == "evaluate":
        config = get_config(dataset=DATASET, model=MODEL)
        oracle: SequentialRecommender = generate_model(config)
        interactions = interaction_generator(config)
        datasets = TimedGenerator(dataset_generator(config=config, use_cache=use_cache))
        evaluate_trace_disalignment(interactions, datasets, oracle)
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
    
