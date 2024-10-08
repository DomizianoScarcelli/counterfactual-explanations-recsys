from sys import orig_argv
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Union
import random
import itertools
from tqdm import tqdm
import torch.nn.functional as F
import torch
from enum import Enum

# A function to calculate distance between two sequences (e.g., edit distance)
def edit_distance(seq1, seq2):
    return 1 - np.sum(np.array(seq1) == np.array(seq2)) / len(seq1)  # Fraction of matching elements

def simple_matching_distance(v1, v2):
    """
    Compute the Simple Matching Distance (SMD) between two discrete integer vectors,
    with equal weights for all elements.
    
    Args:
    - v1 (torch.Tensor): First discrete int vector.
    - v2 (torch.Tensor): Second discrete int vector.
    
    Returns:
    - torch.Tensor: SMD score with equal weights.
    """
    # Ensure the input vectors are of the same shape
    assert v1.shape == v2.shape, "Vectors must have the same shape."
    matches = (v1 == v2).float()
    return matches.mean()

# A function to calculate label distance (0 if same label, 1 if different)
def label_distance(prob1, prob2):
    label1, label2 = prob1.argmax(-1), prob2.argmax(-1)
    return 0 if label1 == label2 else 1

def cosine_distance(prob1: torch.Tensor, prob2: torch.Tensor) -> float:
    return 1 - F.cosine_similarity(prob1, prob2, dim=-1).item()

def self_indicator(seq1, seq2):
    return float("inf") if (seq1 == seq2).all() else 0

# def dummy_model(seq: torch.Tensor) -> torch.Tensor:
    #NOTE: dummy black box model
    return torch.randn((1, 1983))

class GenerationStrategy(ABC):
    @abstractmethod
    def generate(self, x: np.ndarray,  model: Callable[[np.ndarray], torch.Tensor]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pass

class RandomPickGenerationStrategy(GenerationStrategy):
    def generate(self, x: np.ndarray,  model: Callable[[np.ndarray], torch.Tensor]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pass
    
class GeneticGenerationStrategy(GenerationStrategy):
    def __init__(self) :
        self.mutations = [self.mutate_replace, self.mutate_shuffle, self.mutate_swap, self.mutate_reverse]

    # Crossover: Two-point crossover between two sequences
    def two_point_crossover(self, seq1: torch.Tensor, seq2: torch.Tensor):
        assert len(seq1) == len(seq2), "Sequences must be of equal length for crossover."
        child1, child2 = seq1.clone().tolist(), seq2.clone().tolist()
        i, j = sorted(random.sample(range(len(seq1)), 2))
        child1[i:j+1], child2[i:j+1] = seq2[i:j+1], seq1[i:j+1]
        return torch.tensor(child1), torch.tensor(child2)

    def one_point_crossover(self, seq1: torch.Tensor, seq2: torch.Tensor):
        assert len(seq1) == len(seq2), "Sequences must be of equal length for crossover."
        child1, child2 = seq1.clone().tolist(), seq2.clone().tolist()
        i =  random.sample(range(len(seq1)), 1)[0]
        child1[i:], child2[i:] = seq2[i:], seq1[i:]
        return torch.tensor(child1), torch.tensor(child2)
    
    #--------START MUTATIONS--------#
    
    def random_points_with_offset(self, max_value: int, max_offset: int):
        i = random.randint(1, max_value - 1)
        j = random.randint(max(0, i - max_offset), min(max_value - 1, i + max_offset))
        # Sort i and j to ensure i <= j
        return tuple(sorted([i, j]))

    # Mutation: Swap two random elements in the sequence
    def mutate_swap(self, sequence: torch.Tensor, offset_ratio: float=0.8):
        seq = sequence.clone().squeeze(0).tolist()
        max_offset = round(len(seq) * offset_ratio)
        i, j = self.random_points_with_offset(len(seq)-1, max_offset)
        seq[i], seq[j] = seq[j], seq[i]
        return torch.tensor(seq)

    # Mutation: Reverse a random subsequence
    def mutate_reverse(self, sequence: torch.Tensor, offset_ratio:float=0.3):
        seq = sequence.clone().squeeze(0).tolist()
        max_offset = round(len(seq) * offset_ratio)
        i, j = self.random_points_with_offset(len(seq)-1, max_offset)
        seq[i:j+1] = seq[i:j+1][::-1]
        return torch.tensor(seq)
    
    # Mutation: Shuffles a random subsequence
    def mutate_shuffle(self, sequence: torch.Tensor, offset_ratio:float=0.3):
        seq = sequence.clone().squeeze(0).tolist()
        max_offset = round(len(seq) * offset_ratio)
        i, j = self.random_points_with_offset(len(seq)-1, max_offset)
        subseq = seq[i:j+1]  
        random.shuffle(subseq) 
        seq[i:j+1] = subseq  
        return torch.tensor(seq)

    class NumItems(Enum):
        ML_100K=1682
        ML_1M= 3952 - 500

    # Mutation: Replaces an item with another random item
    def mutate_replace(self, sequence, max_value:NumItems=NumItems.ML_1M, num_replaces:int=1):
        seq = sequence.clone().squeeze(0).tolist()
        for _ in range(num_replaces):
            i = random.sample(range(len(seq)), 1)[0]
            new_value = random.randint(1, max_value.value)
            seq[i] = new_value
        return torch.tensor(seq)

    #--------END MUTATIONS--------#


    # Fitness function combining sequence similarity and label difference
    def fitness(self, 
                original_seq: torch.Tensor, 
                candidate_seq: torch.Tensor, 
                original_prob: torch.Tensor, 
                candidate_prob: torch.Tensor,
                good_examples: bool):
        ALPHA1= 0.5
        ALPHA2 = 1 - ALPHA1
        seq_dist = edit_distance(original_seq, candidate_seq) #[0,1]
        label_dist = cosine_distance(original_prob, candidate_prob) #[0,1]
        self_ind = self_indicator(original_seq, candidate_seq) #0 if different, inf if equal
        if not good_examples:
            label_dist = 1 - label_dist
        return ALPHA1 * seq_dist + ALPHA2 * label_dist + self_ind

    # Function to generate initial population of mutated sequences
    def init_population(self, original_seq: torch.Tensor, pop_size: int):
        population = [original_seq for _ in range(pop_size)]
        return population
    
    def mutate(self, population: List[torch.Tensor], pm: float):
        mutated = random.sample(population, round(len(population) * pm))
        for i in range(len(mutated)):
            mutated[i] = random.choice(self.mutations)(mutated[i])
        return population + mutated

    def select(self, population: List[torch.Tensor], fitness_scores: List[float], n: int):
        # ascending order, the lower the better
        population = [population[i] for i in np.argsort([s for s in fitness_scores])][:n]
        return population

    def crossover(self, population: List[torch.Tensor], pc: float):
        # Crossover: generate new population from pairs of parents
        crossover_size = round(len(population) * pc)
        crossover_population = random.sample(population, crossover_size)
        
        # Optimized untouched population: use tensor addresses for quick lookup
        crossover_ids = {id(tensor) for tensor in crossover_population}
        untouched_population = [pop for pop in population if id(pop) not in crossover_ids]
        
        next_population = []
        random.shuffle(crossover_population)  # Shuffle to avoid repetitive sampling
        
        while len(next_population) < crossover_size:
            # Sampling unique pairs without repetition
            for i in range(0, crossover_size - 1, 2):
                parent1, parent2 = crossover_population[i], crossover_population[i+1]
                child1, child2 = self.two_point_crossover(parent1, parent2)
                next_population.extend([child1, child2])
                if len(next_population) >= crossover_size:
                    break
        
        return next_population[:crossover_size] + untouched_population

    def evaluate_fitness(self, original_seq, original_prob, population: List[torch.Tensor], label_func: Callable, good_examples: bool):
        fitness_scores = []
        for candidate_seq in population:
            candidate_prob = label_func(candidate_seq)  # Function to assign label based on the recommender system
            fitness = self.fitness(original_seq, candidate_seq, original_prob, candidate_prob, good_examples)
            fitness_scores.append(fitness)
        return fitness_scores

    # Main genetic algorithm loop
    def genetic_algorithm(self, 
                          original_seq: torch.Tensor,
                          original_prob: torch.Tensor, 
                          label_func: Callable, 
                          pop_size: int=500,
                          generations: int=20, 
                          good_examples: bool=True, 
                          pc: float=0.7, 
                          pm: float=0.5):
        population = self.init_population(original_seq, pop_size)
        for _ in tqdm(range(generations), "Genetic algorithm..."):
            new_population = self.crossover(population, pc=pc)
            # print(f"New population size:", len(new_population))
            mutated_population = self.mutate(new_population, pm=pm)
            # print(f"Mutated population size:", len(mutated_population))
            fitness_scores = self.evaluate_fitness(original_seq=original_seq, original_prob=original_prob, population=mutated_population, label_func=label_func, good_examples=good_examples)
            # print(f"Fitness scores size:", len(fitness_scores))
            selected_population = self.select(mutated_population, fitness_scores, n=pop_size)
            # print(f"Selected population size:", len(selected_population))
            population = selected_population

        labels = [label_func(x).argmax(-1) for x in population]
        return [(pop, lab) for (pop, lab) in zip(population, labels)]
    
    def evaluate_generation(self, examples, label, original_seq):
        # Evaluate label
        same_label = sum(1 for ex in examples if ex[1] == label)
        # Evaluate example similarity
        distances = []
        for seq, _ in examples:
            distances.append(edit_distance(original_seq, seq))
        return (same_label / len(examples)), (sum(distances)/len(distances))

    def generate(self, x: torch.Tensor, model: Callable, clean: bool=False, balance: bool=True): 
        original_prob = model(x)
        original_label = original_prob.argmax(-1).item()
        # print(f"Original label is: {original_prob.argmax(-1).item()}") 

        good_examples = self.genetic_algorithm(original_seq=x, label_func=model, original_prob=original_prob, good_examples=True)
        label_eval, seq_eval = self.evaluate_generation(good_examples, original_label, original_seq=x.squeeze(0))
        print(f"Good examples ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}")

        bad_examples = self.genetic_algorithm(original_seq=x, label_func=model, original_prob=original_prob, good_examples=False)
        label_eval, seq_eval = self.evaluate_generation(bad_examples, original_label, original_seq=x.squeeze(0))
        print(f"Bad examples ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}")
        if clean:
            good_examples = self.clean(good_examples, original_label, good_examples=True)
            bad_examples = self.clean(bad_examples, original_label, good_examples=False)
            label_eval, seq_eval = self.evaluate_generation(good_examples, original_label, original_seq=x.squeeze(0))
            print(f"[After clean] Good examples ({len(good_examples)}) ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}")
            label_eval, seq_eval = self.evaluate_generation(bad_examples, original_label, original_seq=x.squeeze(0))
            print(f"[After clean] Bad examples ({len(bad_examples)}) ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}")
        if balance:
            min_examples = min(len(good_examples), len(bad_examples))
            good_examples, bad_examples = good_examples[:min_examples], bad_examples[:min_examples]
        return good_examples, bad_examples

    def clean(self, examples, label, good_examples: bool):
        if good_examples:
            return [ex for ex in examples if ex[1] == label]
        return [ex for ex in examples if ex[1] != label]


def generate(strategy: GenerationStrategy):
    src = np.random.randint(0,100, (20,))
    good_examples, bad_examples = strategy.generate(src, dummy_model)
    print([ex[1].item() for ex in good_examples], [ex[1].item() for ex in bad_examples])
    print(len(good_examples), len(bad_examples))
    pass



if __name__ == "__main__":
    generate(strategy=GeneticGenerationStrategy())
