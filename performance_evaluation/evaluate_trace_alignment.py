import json
import time
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from constants import MAX_LENGTH
from exceptions import CounterfactualNotFound, DfaNotAccepting, DfaNotRejecting
from graph_search import print_action
from models.ExtendedBERT4Rec import ExtendedBERT4Rec
from recommenders.generate_dataset import (dataset_generator, generate_model,
                                           get_config,
                                           get_sequence_from_interaction,
                                           interaction_generator)
from recommenders.utils import pad_zero, trim_zero
from run import single_run, timed_learning_pipeline, timed_trace_disalignment
from type_hints import RecDataset, RecModel
from utils import TimedGenerator, set_seed

set_seed()

def save_log(log: List[Dict], 
             original: List[int], 
             alignment:Optional[List[str]], 
             status: str, 
             cost: int, 
             time_dataset_generation: float,
             time_automata_learning: float, 
             time_alignment: float):
    """ 
    Saves the log with the evaluation information on the disk

    Args:
        log: [TODO:description]
        original: [TODO:description]
        alignment: [TODO:description]
        status: [TODO:description]
        cost: [TODO:description]
        time_dataset_generation: [TODO:description]
        time_automata_learning: [TODO:description]
        time_alignment: [TODO:description]
    """
    path = "evaluation_log.json"
    original_tostr = ", ".join(str(c) for c in original)
    alignment_tostr = ", ".join(alignment) if alignment else alignment
    total_time =  time_dataset_generation + time_automata_learning + time_alignment
    info = {"original": original_tostr, 
            "alignment": alignment_tostr, 
            "status":status, 
            "cost": cost, 
            "times": {"time_dataset_generation": time_dataset_generation,
                      "time_automata_learning": time_automata_learning,
                      "time_alignment": time_alignment, 
                      "total_time": total_time}
            }

    log.append(info)
    with open(path, "w") as f:
        json.dump(log, f)
    print("Log saved!")
    return log

def evaluate_stats(log_path: str):
    """
    Given an evaluation log path, it returns the stats of the evaluation.

    Args:
        log_path: [TODO:description]
    """
    pass

def evaluate_trace_disalignment(interactions, 
                                datasets, 
                                oracle,
                                num_counterfactuals: int=20):
    good, bad, not_found, skipped = 0, 0, 0, 0
    evaluation_log = []
    status = "unknown"
    for i, ((train, _), interaction) in enumerate(tqdm(zip(datasets, interactions), desc="Performance evaluation...")):
        if i == num_counterfactuals:
            print(f"Generated {num_counterfactuals}, exiting...")
            break
        source_sequence = get_sequence_from_interaction(interaction)
        source_gt = oracle.full_sort_predict(source_sequence).argmax(-1).item()
        source_sequence = trim_zero(source_sequence.squeeze(0)).tolist()
        print(f"Source sequence:", source_sequence)
        time_dataset_generation = datasets.get_times()[i]
        try:
            aligned, cost, alignment = single_run(source_sequence, train)
        except CounterfactualNotFound:
            not_found += 1
            evaluation_log = save_log(evaluation_log, original=source_sequence,
                                      alignment=None,
                                      status="CounterfactualNotFound", cost=0,
                                      time_dataset_generation=time_dataset_generation,
                                      time_automata_learning=0,
                                      time_alignment=0)
            print(f"Counterfactual not found")
            continue
        except DfaNotAccepting as e:
            print(e)
            skipped += 1
            evaluation_log = save_log(evaluation_log, original=source_sequence,
                                      alignment=None,
                                      status="DfaNotAccepting", cost=0,
                                      time_dataset_generation=time_dataset_generation,
                                      time_automata_learning=0,
                                      time_alignment=0)
            continue
        except DfaNotRejecting as e:
            print(e.with_traceback)
            skipped += 1
            evaluation_log = save_log(evaluation_log, original=source_sequence,
                                      alignment=None,
                                      status="DfaNotRejecting", cost=0,
                                      time_dataset_generation=time_dataset_generation,
                                      time_automata_learning=0,
                                      time_alignment=0)
            continue

        if len(aligned) == MAX_LENGTH:
            aligned = torch.tensor(aligned).unsqueeze(0).to(torch.int64)
        elif len(aligned) < MAX_LENGTH:
            aligned = pad_zero(trim_zero(torch.tensor(aligned)), MAX_LENGTH).unsqueeze(0).to(torch.int64)
        else:
            skipped += 1
            evaluation_log = save_log(evaluation_log, original=source_sequence,
                                      alignment=alignment,
                                      status="MaximumLengthReached", 
                                      cost=cost,
                                      time_dataset_generation=time_dataset_generation,
                                      time_automata_learning=timed_learning_pipeline.get_last_time(),
                                      time_alignment=timed_trace_disalignment.get_last_time())
            continue
        print(f"Alignment:", [print_action(a) for a in alignment])
        aligned_gt = oracle.full_sort_predict(aligned).argmax(-1).item()
        correct = source_gt != aligned_gt
        # assert correct, "Source and aligned have the same label {source_gt} == {aligned_gt}"
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
                                  status=status, 
                                  cost=cost,
                                  time_dataset_generation=time_dataset_generation,
                                  time_automata_learning=timed_learning_pipeline.get_last_time(),
                                  time_alignment=timed_trace_disalignment.get_last_time())


if __name__ == "__main__":
    config = get_config(dataset=RecDataset.ML_1M, model=RecModel.BERT4Rec)
    oracle: ExtendedBERT4Rec = generate_model(config)
    interactions = interaction_generator(config)
    datasets = TimedGenerator(dataset_generator(config=config, use_cache=False))
    evaluate_trace_disalignment(interactions, datasets, oracle)
