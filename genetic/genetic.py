import random
from typing import Callable, List

import numpy as np
import torch
from deap import base, creator, tools
from torch import Tensor

from config import GENERATIONS, POP_SIZE
from constants import MAX_LENGTH, MIN_LENGTH
from genetic.extended_ea_algorithms import eaSimpleBatched
from genetic.mutations import (ALL_MUTATIONS, Mutations, mutate_add,
                               mutate_delete)
from genetic.utils import (cosine_distance, cPickle_clone, edit_distance,
                           self_indicator)
from models.utils import pad, pad_batch, trim
from type_hints import Dataset
from utils import set_seed

set_seed()

class GeneticGenerationStrategy():
    def __init__(self, input_seq: Tensor, 
                 predictor: Callable,
                 allowed_mutations: List[Mutations] = ALL_MUTATIONS, 
                 pop_size: int=POP_SIZE, 
                 generations: int=GENERATIONS, 
                 good_examples: bool=True,
                 halloffame_ratio: float=0.1,
                 verbose: bool=True):
        self.input_seq = trim(input_seq)
        self.predictor = predictor
        self.pop_size = pop_size
        self.gt = self.predictor(input_seq.unsqueeze(0))
        self.generations = generations
        self.good_examples = good_examples
        self.halloffame_ratio = halloffame_ratio
        self.allowed_mutations = allowed_mutations
        self.verbose = verbose
        # Define the evaluation function
        creator.create("fitness", base.Fitness, weights=(-1.0,))  # Minimize fitness
        creator.create("individual", list, fitness=creator.fitness)

        self.toolbox = base.Toolbox()
        self.toolbox.register("feature_values", lambda x: x.tolist(), self.input_seq)
        self.toolbox.register("individual", tools.initIterate, creator.individual, self.toolbox.feature_values)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, n=self.pop_size)
        self.toolbox.register("clone", cPickle_clone)

        self.toolbox.register("evaluate", self.evaluate_fitness_batch)
        self.toolbox.register("mate", tools.cxTwoPoint)  # Use two-point crossover
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection
    
    def print(self, s):
        if self.verbose:
            print(s)

    def mutate(self, seq: List[int]):
        #TODO: remove for efficiency
        assert -1 not in seq, f"Seq must not contain padding char: {seq}"
        mutations = self.allowed_mutations.copy()
        if not len(seq) < MAX_LENGTH and mutate_add in mutations:
            mutations.remove(mutate_add)
        if not len(seq) > MIN_LENGTH and mutate_delete in mutations:
            mutations.remove(mutate_delete)
        mutation = random.choice(mutations)
        return mutation(seq)

    def evaluate_fitness_batch(self, individuals: List[List[int]]) -> List[float]:
        #TODO: add a batch_size mechanism
        ALPHA1= 0.5
        ALPHA2 = 1 - ALPHA1
        candidate_seqs = torch.stack([pad(torch.tensor(i), MAX_LENGTH) for i in individuals])
        batch_size = candidate_seqs.size(0)
        candidate_probs = self.predictor(candidate_seqs)  # Function to assign label based on the recommender system
        assert candidate_probs.size(0) == batch_size, f"Mismatch in probs shape and batch size: {candidate_probs.shape} != {batch_size}"
        fitnesses = []
        for batch_idx in range(batch_size):
            candidate_seq = candidate_seqs[batch_idx]
            candidate_prob = candidate_probs[batch_idx]
            seq_dist = edit_distance(self.input_seq, candidate_seq) #[0,1]
            label_dist = cosine_distance(self.gt, candidate_prob) #[0,1]
            self_ind = self_indicator(self.input_seq, candidate_seq) #0 if different, inf if equal
            if not self.good_examples:
                label_dist = 1 - label_dist
            cost = ALPHA1 * seq_dist + ALPHA2 * label_dist + self_ind,
            fitnesses.append(cost)
        return fitnesses


    def generate(self) -> Dataset:
        population = self.toolbox.population(n=self.pop_size)
        
        halloffame_size = int(np.round(self.pop_size * self.halloffame_ratio))
        halloffame = tools.HallOfFame(halloffame_size)

        population, _ = eaSimpleBatched(population, 
                                        self.toolbox, 
                                        cxpb=0.7,
                                        mutpb=0.5, 
                                        ngen=self.generations,
                                        halloffame=halloffame if self.halloffame_ratio != 0 else None,
                                        verbose=False)

        preds = self.predictor(pad_batch(population, MAX_LENGTH)).argmax(-1)
        new_population = [(torch.tensor(x), preds[i].item()) for (i, x) in enumerate(population)]
        label_eval, seq_eval = self.evaluate_generation(new_population)
        self.print(f"[Original] Good examples = {self.good_examples} [{len(new_population)}] ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}")
        if not self.good_examples or self.halloffame_ratio == 0:
            # new_population.append((self.input_seq, self.gt.argmax(-1).item()))
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
        self.print(f"[After clean] Good examples={self.good_examples} ({len(clean_pop)}) ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}")
        return clean_pop

    def evaluate_generation(self, examples):
        # Evaluate label
        label = self.gt.argmax(-1).item()
        same_label = sum(1 for ex in examples if ex[1] == label)
        # Evaluate example similarity
        distances = []
        for seq, _ in examples:
            distances.append(edit_distance(self.input_seq, seq))
        return (same_label / len(examples)), (sum(distances)/len(distances))

