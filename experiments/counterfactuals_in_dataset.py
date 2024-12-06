import json

import torch

from genetic.dataset.utils import load_dataset
from utils_classes.distances import edit_distance


def counterfactual_in_dataset():
    """

    The experiment consists in analyzing how the "bad" points in the generated
    GoodBadDataset are formed, in order to see if they are worse then the
    counterfactual generated with the trace alignment process.
    """
    # NOTE: for simplicity now , I will get the dataset from the cache. This is
    # kind of a manual process, but it's ok for now.

    cache_dir = "dataset_cache/interaction_0_dataset.pickle"
    _, bad = load_dataset(cache_dir)
    source = torch.tensor([2720, 365, 1634, 1229, 140, 351, 1664, 160, 1534, 1233, 618, 267, 2490, 213, 2483, 89, 273, 665, 352, 222, 2265, 2612, 429, 2492, 2827, 532, 1002, 202, 821, 1615, 1284, 830, 176, 1116, 2626, 23, 415, 1988, 694, 133, 1536, 510, 290, 152, 204, 1034, 1273, 289, 462, 165])
    distances = {}
    def to_str(t):
        return ",".join([str(x) for x in t])
    distances[to_str(source.tolist())] = 0
    for seq, _ in bad:
        distance_n = edit_distance(source, seq.squeeze())
        distance_nn = edit_distance(source, seq.squeeze(), normalized=False)
        distances[to_str(seq.squeeze().tolist())] = (distance_nn, distance_n)
    save_path = "results/counterfactual_in_dataset_0.json"
    with open(save_path, "w") as f:
        json.dump(distances, f, indent=2)
    print(f"Experiment results saved in {save_path}")

if __name__ == "__main__":
   counterfactual_in_dataset() 

    
