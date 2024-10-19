import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Tuple

import _pickle as cPickle
import numpy as np
import torch
import torch.nn.functional as F
from deap import base, creator, tools
from torch import Tensor

from constants import MAX_LENGTH, MIN_LENGTH
from extended_ea_algorithms import eaSimpleBatched
from recommenders.utils import pad_zero, pad_zero_batch, trim_zero
from type_hints import Dataset, GoodBadDataset
from utils import set_seed

set_seed()

class NumItems(Enum):
    ML_100K=1682
    ML_1M=3703
    MOCK=6

def cPickle_clone(x):
    # return deepcopy(x)
    return cPickle.loads(cPickle.dumps(x))

def edit_distance(seq1, seq2):
    if len(seq1) < len(seq2):
        seq1 = pad_zero(seq1, len(seq2))
    if len(seq2) < len(seq1):
        seq2 = pad_zero(seq2, len(seq1))

    return 1 - np.sum(np.array(seq1) == np.array(seq2)) / len(seq1)  # Fraction of matching elements

def cosine_distance(prob1: Tensor, prob2: Tensor) -> float:
    return 1 - F.cosine_similarity(prob1, prob2, dim=-1).item()

def self_indicator(seq1, seq2):
    if len(seq1) != len(seq2):
        return 0
    return float("inf") if (seq1 == seq2).all() else 0

def random_points_with_offset(max_value: int, max_offset: int):
    i = random.randint(1, max_value - 1)
    j = random.randint(max(0, i - max_offset), min(max_value - 1, i + max_offset))
    # Sort i and j to ensure i <= j
    return tuple(sorted([i, j]))

def mutate(seq: List[int]):
    # mutations = [mutate_replace, mutate_swap, mutate_shuffle, mutate_reverse]
    mutations = [mutate_replace, mutate_swap]
    if len(seq) < MAX_LENGTH:
        mutations.append(mutate_add)
    if len(seq) > MIN_LENGTH:
        mutations.append(mutate_delete)
    mutation = random.choice(mutations)
    return mutation(seq)

def mutate_replace(seq: List[int], max_value:NumItems=NumItems.ML_1M, num_replaces:int=1):
    for _ in range(num_replaces):
        i = random.sample(range(len(seq)), 1)[0]
        new_value = random.randint(1, max_value.value)
        while (new_value in seq):
            new_value = random.randint(1, max_value.value)
        seq[i] = new_value
    return seq,

def mutate_swap(seq: List[int], offset_ratio: float=0.8):
    max_offset = round(len(seq) * offset_ratio)
    i, j = random_points_with_offset(len(seq)-1, max_offset)
    seq[i], seq[j] = seq[j], seq[i]
    return seq,

def mutate_reverse(seq: List[int], offset_ratio:float=0.3):
    max_offset = round(len(seq) * offset_ratio)
    i, j = random_points_with_offset(len(seq)-1, max_offset)
    seq[i:j+1] = seq[i:j+1][::-1]
    return seq,

# Mutation: Shuffles a random subsequence
def mutate_shuffle(seq: List[int], offset_ratio:float=0.3):
    max_offset = round(len(seq) * offset_ratio)
    i, j = random_points_with_offset(len(seq)-1, max_offset)
    subseq = seq[i:j+1]  
    random.shuffle(subseq) 
    seq[i:j+1] = subseq  
    return seq,

def mutate_add(seq: List[int], max_value: NumItems=NumItems.ML_1M):
    random_item = random.randint(1, max_value.value)
    while random_item in seq:
        random_item = random.randint(1, max_value.value)
    i = random.sample(range(len(seq)), 1)[0]
    seq.insert(i, random_item)
    return seq,

def mutate_delete(seq: List[int]):
    i = random.sample(range(len(seq)), 1)[0]
    seq.remove(seq[i])
    return seq,

class GeneticGenerationStrategy():
    def __init__(self, input_seq: Tensor, predictor: Callable, pop_size: int=1000, generations: int=20, good_examples: bool=True):
        self.input_seq = trim_zero(input_seq)
        self.predictor = predictor
        self.pop_size = pop_size
        self.gt = self.predictor(input_seq.unsqueeze(0))
        self.generations = generations
        self.good_examples = good_examples
        self.halloffame_ratio = 0.1
        # Define the evaluation function
        creator.create("fitness", base.Fitness, weights=(-1.0,))  # Minimize fitness
        creator.create("individual", list, fitness=creator.fitness)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("feature_values", lambda x: x.tolist(), input_seq)
        self.toolbox.register("individual", tools.initIterate, creator.individual, self.toolbox.feature_values)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, n=self.pop_size)
        self.toolbox.register("clone", cPickle_clone)

        self.toolbox.register("evaluate", self.evaluate_fitness_batch)
        self.toolbox.register("mate", tools.cxTwoPoint)  # Use two-point crossover
        self.toolbox.register("mutate", mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection
    
    def evaluate_fitness(self,individual: List[int]):
        #TODO: delete function
        ALPHA1= 0.5
        ALPHA2 = 1 - ALPHA1
        candidate_seq = torch.tensor(individual)
        candidate_prob = self.predictor(candidate_seq)  # Function to assign label based on the recommender system
        seq_dist = edit_distance(self.input_seq, candidate_seq) #[0,1]
        label_dist = cosine_distance(self.gt, candidate_prob) #[0,1]
        self_ind = self_indicator(self.input_seq, candidate_seq) #0 if different, inf if equal
        if not self.good_examples:
            label_dist = 1 - label_dist
        return ALPHA1 * seq_dist + ALPHA2 * label_dist + self_ind,

    def evaluate_fitness_batch(self, individuals: List[List[int]]) -> List[float]:
        #TODO: add a batch_size mechanism
        ALPHA1= 0.5
        ALPHA2 = 1 - ALPHA1
        candidate_seqs = torch.stack([pad_zero(torch.tensor(i), MAX_LENGTH) for i in individuals])
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

        population, _ = eaSimpleBatched(population, self.toolbox, cxpb=0.7,
                                        mutpb=0.5, ngen=self.generations,
                                        halloffame=halloffame, verbose=False)
        preds = self.predictor(torch.stack([pad_zero(torch.tensor(p), MAX_LENGTH) for p in population])).argmax(-1)
        new_population = [(torch.tensor(x), preds[i].item()) for (i, x) in enumerate(population)]
        label_eval, seq_eval = self.evaluate_generation(new_population)
        print(f"Good examples = {self.good_examples} [{len(new_population)}] ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}")
        if not self.good_examples:
            # Augment only good examples, which are the rarest
            return new_population
        
        augmented = self.augment_pop(population, halloffame)
        new_augmented = []
        preds = self.predictor(pad_zero_batch(augmented, MAX_LENGTH)).argmax(-1)
        for i, x in enumerate(augmented):
            new_augmented.append((torch.tensor(x), preds[i].item()))
        label_eval, seq_eval = self.evaluate_generation(new_augmented)
        print(f"[Augmented] Good examples = {self.good_examples} [{len(new_augmented)}] ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}")
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
        print(f"[After clean] Good examples={self.good_examples} ({len(clean_pop)}) ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}")
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

