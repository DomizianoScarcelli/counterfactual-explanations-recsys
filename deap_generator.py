from deap import base, creator, tools
from extended_ea_algorithms import eaSimpleBatched
import numpy as np
import random
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
import torch.nn.functional as F
from enum import Enum
import _pickle as cPickle

class NumItems(Enum):
    ML_100K=1682
    ML_1M= 3952 - 500

def cPickle_clone(x):
    # return deepcopy(x)
    return cPickle.loads(cPickle.dumps(x))

def edit_distance(seq1, seq2):
    return 1 - np.sum(np.array(seq1) == np.array(seq2)) / len(seq1)  # Fraction of matching elements

def cosine_distance(prob1: torch.Tensor, prob2: torch.Tensor) -> float:
    return 1 - F.cosine_similarity(prob1, prob2, dim=-1).item()

def self_indicator(seq1, seq2):
    return float("inf") if (seq1 == seq2).all() else 0

def random_points_with_offset(max_value: int, max_offset: int):
    i = random.randint(1, max_value - 1)
    j = random.randint(max(0, i - max_offset), min(max_value - 1, i + max_offset))
    # Sort i and j to ensure i <= j
    return tuple(sorted([i, j]))

def mutate(seq: List[int]):
    mutation = random.choice([mutate_replace, mutate_swap, mutate_shuffle, mutate_reverse])
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


class GeneticGenerationStrategy():
    def __init__(self, input_seq: torch.Tensor, predictor: Callable, pop_size: int=1000, generations: int=20, good_examples: bool=True):
        self.input_seq = input_seq
        self.predictor = predictor
        self.pop_size = pop_size
        self.gt = self.predictor(input_seq.unsqueeze(0))
        self.generations = generations
        self.good_examples = good_examples
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
        ALPHA1= 0.5
        ALPHA2 = 1 - ALPHA1
        candidate_seqs = torch.tensor(individuals)
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


    def generate(self):
        population = self.toolbox.population(n=self.pop_size)
        population, _ = eaSimpleBatched(population, self.toolbox, cxpb=0.7, mutpb=0.5, ngen=self.generations, verbose=True)
        new_population = []
        inserted = set()
        # for x in population:
        #     if tuple(x) in inserted:
        #         continue
        #     new_population.append((torch.tensor(x), self.predictor(torch.tensor(x)).argmax(-1).item()))
        #     inserted.add(tuple(x))
        preds = self.predictor(torch.tensor(population)).argmax(-1)
        print(f"[generate] preds shape is {preds.shape}")
        for i, x in enumerate(population):
            new_population.append((torch.tensor(x), preds[i].item()))
        population = new_population
        label_eval, seq_eval = self.evaluate_generation(population)
        print(f"Good examples = {self.good_examples} ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}")
        return population
    
    def clean(self, examples):
        label = self.gt.argmax(-1).item()
        if self.good_examples:
            return [ex for ex in examples if ex[1] == label]
        return [ex for ex in examples if ex[1] != label]

    def postprocess(self, population):
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

