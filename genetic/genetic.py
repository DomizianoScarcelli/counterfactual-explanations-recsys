import random
from copy import deepcopy
from typing import Any, Callable, List

import numpy as np
import torch
from deap import base, creator, tools
from torch import Tensor

from config import DETERMINISM, GENERATIONS, POP_SIZE
from constants import MAX_LENGTH, MIN_LENGTH
from genetic.extended_ea_algorithms import (eaSimpleBatched, indexedCxTwoPoint,
                                            indexedSelTournament)
from genetic.mutations import (ALL_MUTATIONS, AddMutation, DeleteMutation,
                               Mutation, contains_mutation, remove_mutation)
from genetic.utils import (_evaluate_generation, clone, cosine_distance,
                           edit_distance, label_indicator, self_indicator,
                           kl_divergence)
from models.utils import pad, pad_batch, trim
from type_hints import Dataset
from utils import set_seed


class GeneticGenerationStrategy():
    def __init__(self, input_seq: Tensor, 
                 predictor: Callable,
                 alphabet: List[int],
                 allowed_mutations: List[Mutation] = ALL_MUTATIONS, 
                 pop_size: int=POP_SIZE, 
                 generations: int=GENERATIONS, 
                 good_examples: bool=True,
                 halloffame_ratio: float=0.1,
                 verbose: bool=True):
        set_seed()
        self.input_seq = trim(input_seq)
        self.predictor = predictor
        self.pop_size = pop_size
        self.gt = self.predictor(input_seq.unsqueeze(0)).squeeze()
        self.generations = generations
        self.good_examples = good_examples
        self.halloffame_ratio = halloffame_ratio
        self.allowed_mutations = allowed_mutations
        self.verbose = verbose
        self.alphabet = alphabet
        # Define the evaluation function
        creator.create("fitness", base.Fitness, weights=(-1.0,))  # Minimize fitness
        creator.create("individual", list, fitness=creator.fitness)

        self.toolbox: Any = base.Toolbox()
        self.toolbox.register("feature_values", lambda x: x.tolist(), self.input_seq)
        self.toolbox.register("individual", tools.initIterate, creator.individual, self.toolbox.feature_values)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, n=self.pop_size)
        self.toolbox.register("clone", clone)

        self.toolbox.register("evaluate", self.evaluate_fitness_batch)
        self.toolbox.register("mate", indexedCxTwoPoint)  # Use two-point crossover
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", indexedSelTournament, tournsize=3)  # Tournament selection
    
    def print(self, s):
        if self.verbose:
            print(s)

    def mutate(self, seq: List[int], index: int):
        # Set seed according to the index in order to always choose a different mutation
        set_seed(index)
        #TODO: remove for efficiency
        assert -1 not in seq, f"Seq must not contain padding char: {seq}"
        mutations = self.allowed_mutations.copy()
        if not len(seq) < MAX_LENGTH and contains_mutation(AddMutation, mutations):
            mutations = remove_mutation(AddMutation, mutations)
        if not len(seq) > MIN_LENGTH and contains_mutation(DeleteMutation, mutations):
            mutations = remove_mutation(DeleteMutation, mutations)
        mutation = random.choice(mutations)
        result = mutation(deepcopy(seq), self.alphabet, index)
        set_seed()
        return result

    def evaluate_fitness_batch(self, individuals: List[List[int]]) -> List[float]:
        #TODO: add a batch_size mechanism
        set_seed()
        ALPHA1= 0.5 
        ALPHA2 = 1 - ALPHA1
        candidate_seqs = pad_batch(individuals, MAX_LENGTH)
        batch_size = candidate_seqs.size(0)
        # Function to assign label based on the recommender system
        candidate_probs = self.predictor(candidate_seqs)  
        assert candidate_probs.size(0) == batch_size, f"Mismatch in probs shape and batch size: {candidate_probs.shape} != {batch_size}"
        fitnesses = []

        test_mapping = {}
        for batch_idx in range(batch_size):
            set_seed()
            candidate_seq = trim(candidate_seqs[batch_idx])
            candidate_prob = candidate_probs[batch_idx]
            
            ##NOTE: this is a test
            #if DETERMINISM:
            #    gt = candidate_prob.argmax(-1).item()
            #    if tuple(candidate_seq.tolist()) in test_mapping:
            #        cached = test_mapping[tuple(candidate_seq.tolist())]
            #        assert gt == cached, f"Label are different: {gt} != {cached}"
            #    else:
            #        test_mapping[tuple(candidate_seq.tolist())] = gt

            assert self.gt.shape == candidate_prob.shape
            seq_dist = edit_distance(self.input_seq, candidate_seq) #[0,MAX_LENGTH] if not normalized, [0,1] if normalized
            label_dist = label_indicator(candidate_prob, self.gt)
            self_ind = self_indicator(self.input_seq, candidate_seq) #0 if different, inf if equal
            # if self.gt.argmax(-1).item() != candidate_prob.argmax(-1).item():
            #     print(f""" 
            #           [DEBUG]
            #           seq_dist: {seq_dist}
            #           label_dist: {label_dist}
            #           self_ind: {self_ind}
            #           ---
            #           input_seq: {self.input_seq}
            #           candidate_seq: {candidate_seq}
            #           ---
            #           gt shape: {self.gt.shape}
            #           candidate_prob shape: {candidate_prob.shape}
            #           gt: {self.gt}
            #           candidate_prob : {candidate_prob}
            #           ---
            #           gt.item(): {self.gt.argmax(-1).item()}
            #           candidate_prob.item(): {candidate_prob.argmax(-1).item()}
            #           """)
            if not self.good_examples:
                # label_dist = 0 if label_dist == float("inf") else float("inf")
                label_dist = 1 - label_dist
            cost = ALPHA1 * seq_dist + ALPHA2 * label_dist + self_ind,
            fitnesses.append(cost)

        # print(f"[DEBUG] Fitnesses: {list(sorted(fitnesses))}")
        return fitnesses


    def generate(self) -> Dataset:
        set_seed()
        population = self.toolbox.population()
        
        halloffame_size = int(np.round(self.pop_size * self.halloffame_ratio))
        halloffame = tools.HallOfFame(halloffame_size)

        population, _ = eaSimpleBatched(population, 
                                        self.toolbox, 
                                        cxpb=0.7,
                                        mutpb=0.5, 
                                        ngen=self.generations,
                                        halloffame=halloffame if self.halloffame_ratio != 0 else None,
                                        verbose=False,
                                        pbar=self.verbose)
        preds = self.predictor(pad_batch(population, MAX_LENGTH)).argmax(-1)
        new_population = [(torch.tensor(x), preds[i].item()) for (i, x) in enumerate(population)]
        label_eval, seq_eval = self.evaluate_generation(new_population)
        self.print(f"[Original] Good examples = {self.good_examples} [{len(new_population)}] ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}")
        if not self.good_examples or self.halloffame_ratio == 0:
            # Augment only good examples, which are the rarest
            return new_population
        
        augmented = self.augment_pop(population, halloffame)
        new_augmented = []
        preds = self.predictor(pad_batch(augmented, MAX_LENGTH)).argmax(-1)
        for i, x in enumerate(augmented):
            new_augmented.append((torch.tensor(x), preds[i].item()))
        label_eval, seq_eval = self.evaluate_generation(new_augmented)
        self.print(f"[Augmented] Good examples = {self.good_examples} [{len(new_augmented)}] ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}")
        return new_augmented

    def augment_pop(self, population, halloffame):
        fitness_values = [p.fitness.wvalues[0] for p in population if p.fitness.wvalues[0] != float("-inf")]
        fitness_values = sorted(fitness_values)
        fitness_diff = [fitness_values[i+1] - fitness_values[i] for i in range(0, len(fitness_values)-1)]

        index = np.max(np.argwhere(fitness_diff == np.amax(fitness_diff)).flatten().tolist())
        fitness_value_thr = fitness_values[index]
        
        oversample = list()
        
        for p in population:
            if p.fitness.wvalues[0] > fitness_value_thr:
                oversample.append(list(p))
                
        for h in halloffame:
            if h.fitness.wvalues[0] > fitness_value_thr:
                oversample.append(list(h))
                
        return oversample
    
    def clean(self, examples: Dataset) -> Dataset:
        label = self.gt.argmax(-1).item()
        if self.good_examples:
            return [ex for ex in examples if ex[1] == label]
        return [ex for ex in examples if ex[1] != label]

    def postprocess(self, population: Dataset) -> Dataset:
        clean_pop = self.clean(population)
        label_eval, seq_eval = self.evaluate_generation(clean_pop)

        source_point = (self.input_seq, self.gt.argmax(-1).item())

        # Remove any copy of the source point from the good or bad dataset.
        new_pop = [(ind, label) for ind, label in clean_pop if ind.tolist() != source_point[0].tolist()]
        if len(new_pop) < len(clean_pop):
            self.print(f"Source point was in the dataset {len(clean_pop) - len(new_pop)} times!, removing it")
        clean_pop = new_pop

        # If source point is not in good datset, add just one instance back
        if self.good_examples and len([ind for ind, _ in clean_pop if ind.tolist() == source_point[0].tolist()]) == 0:
            clean_pop.append(source_point)

        label_eval, seq_eval = self.evaluate_generation(clean_pop)
        self.print(f"[After clean] Good examples={self.good_examples} ({len(clean_pop)}) ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}")
        return clean_pop

    def evaluate_generation(self, examples):
        return _evaluate_generation(self.input_seq, examples, self.gt.argmax(-1).item())

