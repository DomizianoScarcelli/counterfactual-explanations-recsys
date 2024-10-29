import json
from statistics import mean
from typing import Dict, List, Optional, Tuple

from type_hints import Split
import time
import datetime


def save_log(log: Dict[str, List[Dict]], 
             original: List[int], 
             alignment:Optional[List[str]], 
             splits_key: str,
             status: str, 
             cost: int, 
             time_dataset_generation: float,
             time_automata_learning: float, 
             time_alignment: float,
             use_cache: bool):
    """ 
    Saves the log with the evaluation information on the disk
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
                      "total_time": total_time},
            "cached": use_cache,
            "timestamp": datetime.datetime.fromtimestamp(time.time()).isoformat()
            }

    log[splits_key].append(info)
    with open(path, "w") as f:
        json.dump(log, f)
    print("Log saved!")
    return log

def evaluate_stats(log_path: str, output_path: str):
    """
    Given an evaluation log path, it returns the stats of the evaluation.

    Args:
        log_path: The path of the evaluation log to load.
    """
    with open(log_path) as f:
        evaluation_log = json.load(f)
    
    result = {}
    for split in evaluation_log:
        stats = {"total_runs": 0, "good_runs": 0, "bad_runs":
                 {"counterfactual_not_found": 0, "malformed_dfa": 0, "other": 0},
                 "mean_time_dataset_generation": 0, "mean_time_automata_learning":
                 0, "mean_time_alignment": 0, "mean_total_time": 0, "min_cost": 0,
                 "max_cost": 0, "mean_cost": 0}
        costs = []
        for run in evaluation_log[split]:
            stats["total_runs"] += 1
            if run["status"] == "good":
                stats["good_runs"] += 1
            elif run["status"] == "bad":
                stats["bad_runs"]["counterfactual_not_found"] += 1
            elif run["status"] == "DfaNotRejecting" or run["status"] == "DfaNotAccepting":
                stats["bad_runs"]["malformed_dfa"] += 1
            elif run["status"] == "skipped":
                stats["bad_runs"]["other"] += 1
            
            if run["status"] == "good":
                stats["mean_time_dataset_generation"] += run["times"]["time_dataset_generation"]
                stats["mean_time_automata_learning"] += run["times"]["time_automata_learning"]
                stats["mean_time_alignment"] += run["times"]["time_alignment"]
                stats["mean_total_time"] += run["times"]["total_time"]

                costs.append(run["cost"])
        stats["mean_time_dataset_generation"] /= stats["good_runs"]
        stats["mean_time_automata_learning"] /= stats["good_runs"]
        stats["mean_time_alignment"] /= stats["good_runs"]
        stats["mean_total_time"] /= stats["good_runs"]

        stats["min_cost"] = min(costs)
        stats["max_cost"] = max(costs)
        stats["mean_cost"] = mean(costs)
        
        result[split] = stats

    with open(output_path, "w") as f:
        json.dump(result, f)

    return result
