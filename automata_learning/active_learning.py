from sklearn.metrics.pairwise import normalize
from automata_learning.passive_learning import generate_automata_from_dataset
from generation.mutations import ALL_MUTATIONS, parse_mutations
from generation.utils import labels2cat
import random
from models.config_utils import generate_model, get_config
from constants import MIN_LENGTH, cat2id
from models.utils import topk
from models.utils import pad
from aalpy.oracles import RandomWalkEqOracle
from aalpy.SULs import SUL
from constants import MAX_LENGTH
from torch.nn import MaxPool1d
from generation.utils import equal_ys, get_items
from aalpy.learning_algs.deterministic.LStar import Oracle, run_Lstar
from recbole.model.abstract_recommender import SequentialRecommender
from utils import printd
import os
import torch
from pathlib import Path
from typing import List, Tuple, Union

from aalpy.automata.Dfa import Dfa
from aalpy.learning_algs import run_RPNI
from torch import Tensor

from alignment.alignment import augment_constraint_automata
from automata_learning.utils import load_automata
from config import ConfigParams
from generation.dataset.utils import load_dataset
from type_hints import GoodBadDataset, RecDataset, RecModel
from utils_classes.distances import edit_distance


class TargetLabelSUL(SUL):
    def __init__(self, model: SequentialRecommender, target: str, k: int):
        super().__init__()
        self.model = model
        self.target = target
        self.k = k
        self.sequence: List[int] = []

    def pre(self):
        self.sequence = []
        pass

    def post(self):
        pass

    def step(self, letter: int):
        if letter is None:
            return False
        self.sequence.append(letter)
        print(f"[DEBUG] self.sequence", self.sequence)
        seq = pad(torch.tensor(self.sequence), MAX_LENGTH).unsqueeze(0)
        preds = self.model(seq)
        topk_preds = topk(logits=preds, dim=-1, k=self.k, indices=True).squeeze()
        topk_preds = labels2cat(topk_preds, encode=True)
        targets = [{cat2id[self.target]} for _ in range(self.k)]
        is_good = equal_ys(topk_preds, targets, return_score=False)
        return is_good


class NeighborhoodEqOracle(Oracle):
    def __init__(self, sul: SUL, alphabet: List[int], seq: List[int]):
        super().__init__(alphabet, sul)
        self.seq = seq
        self.mutations = parse_mutations(ConfigParams.ALLOWED_MUTATIONS)
        self.steps = 100

    def perturbate(self) -> List[int]:
        dist = 0
        max_dist = list(range(1, 6))
        perturbed_seq = self.seq.copy()
        while dist <= random.choice(max_dist):
            mutation = random.choice(self.mutations)
            perturbed_seq = mutation(perturbed_seq, alphabet=self.alphabet)[0]
            print(f"perturbed seq is", perturbed_seq)
            if not MIN_LENGTH <= len(perturbed_seq) <= MAX_LENGTH:
                continue
            dist = edit_distance(self.seq, perturbed_seq, normalized=False)
        return perturbed_seq

    def find_cex(self, hypothesis):
        cexs = []
        for step in range(self.steps):
            pseq = self.perturbate()
            hyp_out = None
            for c in pseq:
                hyp_out = hypothesis.step(c)
            sul_out = self.sul.step(pseq)
            assert hyp_out is not None
            if hyp_out != sul_out:
                cexs.append(pseq)
        return cexs if cexs != [] else None


def generate_automata_from_oracle(
    input_seq: List[int],
    model: SequentialRecommender,
    load_if_exists: bool = True,
    save_path: str = "automata.pickle",
):
    alphabet = list(get_items())
    print(f"[DEBUG] alphabet", alphabet)
    target = "Action"  # TODO: just for debug, change it
    k = 5  # TODO: just for debug, change it
    sul = TargetLabelSUL(model, target, k)
    oracle = NeighborhoodEqOracle(sul=sul, alphabet=alphabet, seq=input_seq)
    dfa = run_Lstar(alphabet=alphabet, sul=sul, eq_oracle=oracle, automaton_type="dfa")
    return dfa


if __name__ == "__main__":
    conf = get_config(dataset=RecDataset.ML_100K, model=RecModel.BERT4Rec)
    model = generate_model(conf)
    input_seq = [
        178,
        290,
        154,
        362,
        322,
        733,
        1007,
        61,
        972,
        141,
        114,
        137,
        1055,
        312,
        348,
        241,
        112,
        53,
        32,
        298,
        321,
        89,
        368,
        492,
        493,
        572,
        356,
        1009,
        69,
        404,
        96,
        720,
        204,
        176,
        428,
        162,
        366,
        182,
        26,
        116,
        453,
        311,
        242,
        286,
        463,
    ]
    dfa = generate_automata_from_oracle(input_seq=input_seq, model=model)
    print(dfa)
