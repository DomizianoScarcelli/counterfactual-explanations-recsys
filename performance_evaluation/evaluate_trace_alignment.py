import json
import time

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
from run import single_run
from type_hints import RecDataset, RecModel
from utils import set_seed

set_seed()

def save_log(log, original, alignment, status: str, cost: int, time: float):
    path = "evaluation_log.json"
    info = {"original": ", ".join(original), "alignment": ", ".join(alignment), "status":status, "cost": cost, "time_to_generate": time}
    log.append(info)
    with open(path, "w") as f:
        json.dump(log, f)
    print("Log saved!", log)
    return log

def evaluate_trace_disalignment(interactions, 
                                datasets, 
                                oracle,
                                num_counterfactuals: int=20):
    good, bad, not_found, skipped = 0, 0, 0, 0
    evaluation_log = []
    status = "unknown"
    for i, (_dataset, interaction) in enumerate(tqdm(zip(datasets, interactions), desc="Performance evaluation...")):
        start = time.time()
        if i == num_counterfactuals:
            print(f"Generated {num_counterfactuals}, exiting...")
            break
        source_sequence = get_sequence_from_interaction(interaction)
        source_gt = oracle.full_sort_predict(source_sequence).argmax(-1).item()
        source_sequence = trim_zero(source_sequence.squeeze(0)).tolist()
        print(f"Source sequence:", source_sequence)
        try:
            aligned, cost, alignment = single_run(source_sequence, _dataset)
        except CounterfactualNotFound:
            not_found += 1
            evaluation_log = save_log(evaluation_log, original=source_sequence, alignment=None, status="not_found", cost=0, time=0)
            print(f"Counterfactual not found")
            continue
        except DfaNotAccepting as e:
            print(e)
            skipped += 1
            continue
        except DfaNotRejecting as e:
            print(e.with_traceback)
            skipped += 1
            continue

        if len(aligned) == MAX_LENGTH:
            aligned = torch.tensor(aligned).unsqueeze(0).to(torch.int64)
        elif len(aligned) < MAX_LENGTH:
            aligned = pad_zero(torch.tensor(aligned), MAX_LENGTH).unsqueeze(0).to(torch.int64)
        else:
            print(f"Aligned sequence exceedes the maximum length that the model is capable of ingesting: {len(aligned)} > {MAX_LENGTH}")
            skipped += 1
            evaluation_log = save_log(evaluation_log, original=source_sequence, alignment=alignment, status="skipped", cost=0, time=0)
            continue
        print(f"Aligned sequence:", aligned.tolist())
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
        print(f"[{i}] Good: {good}, Bad: {bad}, Not Found: {not_found}, Skipped: {skipped} in time {time.time() - start}")
        evaluation_log = save_log(evaluation_log, original=source_sequence, alignment=[print_action(a) for a in alignment], status=status, cost=cost, time=time.time() - start)


if __name__ == "__main__":
    config = get_config(dataset=RecDataset.ML_1M, model=RecModel.BERT4Rec)
    oracle: ExtendedBERT4Rec = generate_model(config)
    interactions = interaction_generator(config)
    datasets = dataset_generator(config=config)
    evaluate_trace_disalignment(interactions, datasets, oracle)
