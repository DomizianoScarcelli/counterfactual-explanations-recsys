import random
from typing import List

import torch
from aalpy.automata.Dfa import Dfa
from aalpy.learning_algs.deterministic.LStar import Oracle, run_Lstar
from aalpy.SULs import SUL
from recbole.model.abstract_recommender import SequentialRecommender

from config.config import ConfigParams
from config.constants import MAX_LENGTH, MIN_LENGTH, cat2id
from core.generation.mutations import parse_mutations
from core.generation.utils import equal_ys, get_items, labels2cat
from core.models.config_utils import generate_model, get_config
from core.models.utils import pad, topk
from type_hints import RecDataset, RecModel
from utils.distances import edit_distance


class TargetLabelSUL(SUL):
    def __init__(self, model: SequentialRecommender, target: str, k: int):
        super().__init__()
        self.model = model
        self.target = target
        self.k = k

    def pre(self):
        self.sequence = []
        pass

    def post(self):
        pass

    def step(self, letter: int):
        if letter is None:
            return False
        self.sequence.append(letter)
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
        self.steps = 300

    def perturbate(self) -> List[int]:
        dist = 0
        max_dist = list(range(1, 2))
        perturbed_seq = self.seq.copy()
        while dist <= random.choice(max_dist):
            mutation = random.choice(self.mutations)
            perturbed_seq = mutation(perturbed_seq, alphabet=self.alphabet)[0]
            if not MIN_LENGTH <= len(perturbed_seq) <= MAX_LENGTH:
                continue
            dist = edit_distance(self.seq, perturbed_seq, normalized=False)
        return perturbed_seq

    def find_cex(self, hypothesis: Dfa):
        for step in range(self.steps):
            pseq = tuple(self.perturbate())
            hyp_out = None
            sul_out = None
            self.sul.pre()
            for c in pseq:
                hyp_out = hypothesis.step(c)
                sul_out = self.sul.step(c)
            assert hyp_out is not None
            assert sul_out is not None
            self.sul.post()
            hypothesis.reset_to_initial()
            if hyp_out != sul_out:
                return pseq
        return None


def generate_automata_from_oracle(
    input_seq: List[int],
    model: SequentialRecommender,
    load_if_exists: bool = True,
    save_path: str = "automata.pickle",
):
    alphabet = list(get_items())
    target = "Action"  # TODO: just for debug, change it
    k = 5  # TODO: just for debug, change it
    sul = TargetLabelSUL(model, target, k)
    oracle = NeighborhoodEqOracle(sul=sul, alphabet=alphabet, seq=input_seq)
    dfa = run_Lstar(
        alphabet=alphabet,
        sul=sul,
        eq_oracle=oracle,
        automaton_type="dfa",
        print_level=3,
        max_learning_rounds=2
    )
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
