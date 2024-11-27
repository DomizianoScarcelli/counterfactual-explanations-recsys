import random
from copy import deepcopy
from typing import Any, List

import pytest
import torch
from deap import base, creator, tools
from tqdm import tqdm

from alignment.alignment import augment_constraint_automata
from automata_learning.learning import (generate_single_accepting_sequence_dfa,
                                        learning_pipeline)
from automata_learning.utils import run_automata
from config import ConfigParams
from constants import MAX_LENGTH, MIN_LENGTH
from genetic.dataset.generate import generate
from genetic.extended_ea_algorithms import (eaSimpleBatched, indexedCxTwoPoint,
                                            indexedSelTournament,
                                            indexedVarAnd)
from genetic.mutations import (AddMutation, DeleteMutation, ReplaceMutation,
                               ReverseMutation, ShuffleMutation, SwapMutation,
                               contains_mutation, remove_mutation)
from genetic.utils import NumItems, clone
from utils_classes.distances import edit_distance, self_indicator, cosine_distance
from models.config_utils import generate_model, get_config
from models.model_funcs import model_predict
from models.utils import pad, pad_batch, trim
from utils import set_seed
from utils_classes.generators import SequenceGenerator


@pytest.mark.heavy
def test_accepting(model, sequences):
    for i, seq in enumerate(sequences):
        if i > 20:
            break
        dataset = generate(seq, model)
        a_dfa = learning_pipeline(seq.squeeze().tolist(), dataset)
        t_dfa = generate_single_accepting_sequence_dfa(seq.squeeze().tolist())
        a_dfa_aug = augment_constraint_automata(a_dfa, t_dfa)
        assert run_automata(a_dfa, seq.squeeze().tolist()), f"Automata does not accept {seq.squeeze().tolist()} at index {i}"
        assert run_automata(a_dfa_aug, seq.squeeze().tolist()), f"Augmented automata does not accept {seq.squeeze().tolist()} at index {i}"

@pytest.mark.heavy
def test_contains_exactly_one_source_sequence(model, sequences):
    """
    Tests if dataset contains only a single reference to the source sequence.
    """
    i = 0
    start_i, end_i = 3, 20
    while True:
        if i < start_i:
            i+=1
            continue
        if i > end_i:
            break
        try:
            seq = next(sequences)
        except StopIteration:
            break
        (good, bad), _ = generate(seq, model)

        count = 0
        # points in good are of shape [50], seq is [1,50]
        seq = pad(seq.squeeze(), MAX_LENGTH)
        # print("Dataset point shape is", good[0][0].shape)
        # print("Source point shape is", seq.shape)
        seq_in_good = sum(torch.all(point == seq) for point, _ in good)
        seq_in_bad = sum(torch.all(point == seq) for point, _ in bad)
        assert seq_in_good == 1, f"[i:{i}] Original sequence must appear EXACTLY ONCE in the good dataset, it appears {count} times"
        assert seq_in_bad == 0, f"[i:{i}] Original sequence should NOT appear in the BAD dataset, it appears {count} times"

        i += 1

class TestGeneticDeterminism:
    #TODO: while this passes each time, when asserting determinism with the `test_mapping` dictionary in the real funciton, it doesn't passes.
    def init_vars(self):
        set_seed()
        config = get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)
        sequences = SequenceGenerator(config)
        self.input_seq = trim(next(sequences).squeeze(0))
        self.model = generate_model(config)
        self.gt = model_predict(self.input_seq.unsqueeze(0), self.model, prob=True).squeeze(0)
        self.source = next(sequences).squeeze(0)
        creator.create("fitness", base.Fitness, weights=(-1.0,))  # Minimize fitness
        creator.create("individual", list, fitness=creator.fitness)
        self.pop_size = 10

        self.toolbox: Any = base.Toolbox()
        self.toolbox.register("feature_values", lambda x: x.tolist(), self.source)
        self.toolbox.register("individual", tools.initIterate, creator.individual, self.toolbox.feature_values)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, n=self.pop_size)
        self.toolbox.register("clone", clone)
        self.toolbox.register("evaluate", self.evaluate_fitness_batch)
        self.toolbox.register("select", indexedSelTournament, tournsize=3)  # Tournament selection
        self.toolbox.register("mate", indexedCxTwoPoint)  # Use two-point crossover
        self.toolbox.register("mutate", self.extracted_mutate)

        self.toolbox2: Any = base.Toolbox()
        self.toolbox2.register("feature_values", lambda x: x.tolist(), self.source)
        self.toolbox2.register("individual", tools.initIterate, creator.individual, self.toolbox.feature_values)
        self.toolbox2.register("population", tools.initRepeat, list, self.toolbox.individual, n=self.pop_size)
        self.toolbox2.register("clone", clone)
        self.toolbox2.register("evaluate", self.evaluate_fitness_batch)
        self.toolbox2.register("select", indexedSelTournament, tournsize=3)  # Tournament selection
        self.toolbox2.register("mate", indexedCxTwoPoint)  # Use two-point crossover
        self.toolbox2.register("mutate", self.extracted_mutate)

        self.allowed_mutations =  [ReplaceMutation(),
                                   SwapMutation(),
                                   ReverseMutation(),
                                   ShuffleMutation(),
                                   AddMutation(),
                                   DeleteMutation()]
        self.alphabet = list(range(0, NumItems.ML_1M.value))

        self.pop1 = self.toolbox.population()
        self.pop2 = self.toolbox2.population()

        self.n_gens = 10

        assert self.pop1 == self.pop2
        assert len(self.pop1) == self.pop_size

        self.good_examples = True

    def evaluate_fitness_batch(self, individuals: List[List[int]]) -> List[float]:
        #TODO: add a batch_size mechanism
        set_seed()
        ALPHA1= 0.5
        ALPHA2 = 1 - ALPHA1
        candidate_seqs = pad_batch(individuals, MAX_LENGTH)
        batch_size = candidate_seqs.size(0)
        candidate_probs = model_predict(candidate_seqs, self.model, prob=True)  
        assert candidate_probs.size(0) == batch_size, f"Mismatch in probs shape and batch size: {candidate_probs.shape} != {batch_size}"
        fitnesses = []

        test_mapping = {}
        for batch_idx in range(batch_size):
            candidate_seq = trim(candidate_seqs[batch_idx])
            candidate_prob = candidate_probs[batch_idx]
            if tuple(candidate_seq.tolist()) in test_mapping:
                gt = candidate_prob.argmax(-1)
                cached = test_mapping[tuple(candidate_seq.tolist())]
                assert gt == cached, f"Label are different: {gt} != {cached}"
            else:
                test_mapping[tuple(candidate_seq.tolist())] = candidate_prob.argmax(-1).item()
            seq_dist = edit_distance(self.input_seq, candidate_seq) #[0,1]
            label_dist = cosine_distance(self.gt, candidate_prob) #[0,1]
            self_ind = self_indicator(self.input_seq, candidate_seq) #0 if different, inf if equal
            if not self.good_examples:
                label_dist = 1 - label_dist
            cost = ALPHA1 * seq_dist + ALPHA2 * label_dist + self_ind,
            fitnesses.append(cost)
        return fitnesses

    def extracted_mutate(self, seq: List[int], index: int):
        #TODO: this is not the most elegant solution, since the real method is in GeneticGenerationStrategy, but I don't want to instantiate it. 
        # a solution would be to extract the mutate  operation into the genetic/utils.py file, and then use it in the GeneticGenerationStrategy.mutate
        set_seed(index)
        mutations = self.allowed_mutations.copy()
        if not len(seq) < MAX_LENGTH and contains_mutation(AddMutation, mutations):
            mutations = remove_mutation(AddMutation, mutations)
        if not len(seq) > MIN_LENGTH and contains_mutation(DeleteMutation, mutations):
            mutations = remove_mutation(DeleteMutation, mutations)
        mutation = random.choice(mutations)
        result = mutation(deepcopy(seq), self.alphabet, index)
        set_seed()
        return result

    def test_evaluate_fitness_batch_determinism(self):
        self.init_vars()
        fitnesses1 = self.evaluate_fitness_batch(self.pop1)
        fitnesses2 = self.evaluate_fitness_batch(self.pop2)

        assert fitnesses1 == fitnesses2

    
   

    def test_indexedVarAnd_determinism(self):
        self.init_vars()
        cxpb = 0.5
        mutpb = 0.5
        offspring1 = indexedVarAnd(self.pop1, self.toolbox, cxpb, mutpb)
        offspring2 = indexedVarAnd(self.pop1, self.toolbox, cxpb, mutpb)
        offspring3 = indexedVarAnd(self.pop2, self.toolbox, cxpb, mutpb)
        assert offspring1 == offspring2 == offspring3
    
    def test_selTorunament_determinism(self):
        self.init_vars()
        chosen1 = indexedSelTournament(self.pop1, self.pop_size // 2, 3)
        chosen2 = indexedSelTournament(self.pop2, self.pop_size // 2, 3)
        assert chosen1 == chosen2

    def test_indexedCxTwoPoint_determinims(self):
        self.init_vars()
        ind1 = self.pop1[0]
        for i in range(100):
            ind1 = self.extracted_mutate(ind1, i)[0]
        ind2 = self.pop1[1]
        assert ind1 != ind2
        _, indices1 = indexedCxTwoPoint(clone(ind1), clone(ind2), 1, return_indices=True)
        _, indices2 = indexedCxTwoPoint(clone(ind1), clone(ind2), 1, return_indices=True)
        _, indices3 = indexedCxTwoPoint(clone(ind1), clone(ind2), 2, return_indices=True)
        assert indices1 == indices2
        assert indices1 != indices3

    def test_mutation_determinism(self):
        """
        Tests if the population is mutated in a deterministic way, meaning for each
        list of sequences, we will always obtain the same list of mutated
        sequences.
        """
        self.init_vars()
        set_seed()
        mutpb = 0.5
        n_gens = 10

        for _ in tqdm(range(n_gens), f"Testing mutation determinism for {n_gens} generation"):
            for i in range(len(self.pop1)):
                if random.random() < mutpb:
                    self.pop1[i], = self.extracted_mutate(clone(self.pop1[i]), i)

            for i in range(len(self.pop2)):
                if random.random() < mutpb:
                    self.pop2[i], = self.extracted_mutate(clone(self.pop2[i]), i)

            assert self.pop1 == self.pop2

        # print()
        # print(self.pop1)
        # print("----------")
        # print(self.pop2)


    def test_eaSimpleBatched_determinism(self):
        self.init_vars()
        set_seed()
        population1, _ = eaSimpleBatched(clone(self.pop1), 
                                        self.toolbox, 
                                        cxpb=0.7,
                                        mutpb=0.5, 
                                        ngen=self.n_gens,
                                        halloffame=None,
                                        verbose=False)
        set_seed()
        population2, _ = eaSimpleBatched(clone(self.pop2), 
                                        self.toolbox2, 
                                        cxpb=0.7,
                                        mutpb=0.5, 
                                        ngen=self.n_gens,
                                        halloffame=None,
                                        verbose=False)

        assert population1 == population2
