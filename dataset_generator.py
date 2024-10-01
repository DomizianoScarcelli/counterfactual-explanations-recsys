import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
import random

# A function to calculate similarity between two sequences (e.g., edit distance)
def sequence_similarity(seq1, seq2):
    return np.sum(np.array(seq1) == np.array(seq2)) / len(seq1)  # Fraction of matching elements

# A function to calculate label distance (0 if same label, 1 if different)
def label_distance(label1, label2):
    return 0 if label1 == label2 else 1

def self_indicator(seq1, seq2):
    return float("inf") if (seq1 == seq2).all() else 0

class GenerationStrategy(ABC):
    @abstractmethod
    def generate(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pass

class RandomPickGenerationStrategy(GenerationStrategy):
    def generate(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pass
    
class GeneticGenerationStrategy(GenerationStrategy):

    # Mutation: Swap two random elements in the sequence
    def mutate_swap(self, sequence):
        seq = sequence.copy()
        i, j = random.sample(range(len(seq)), 2)
        seq[i], seq[j] = seq[j], seq[i]
        return seq

    # Mutation: Reverse a random subsequence
    def mutate_reverse(self, sequence):
        seq = sequence.copy()
        i, j = sorted(random.sample(range(len(seq)), 2))
        seq[i:j+1] = seq[i:j+1][::-1]
        return seq
    
    # Mutation: Shuffles a random subsequence
    def mutate_shuffle(self, sequence):
        return sequence #TODO:
        seq = sequence.copy()
        i, j = sorted(random.sample(range(len(seq)), 2))
        seq[i:j+1] = np.random.shuffle(seq[i:j+1])
        return seq

    # Mutation: Replaces an item with another random item
    def mutate_replace(self, sequence):
        pass


     # Fitness function combining sequence similarity and label difference
    def fitness(self, original_seq, candidate_seq, original_label, candidate_label):
        similarity = sequence_similarity(original_seq, candidate_seq)
        label_dist = label_distance(original_label, candidate_label)
        self_ind = self_indicator(original_seq, candidate_seq)
        return similarity, label_dist, self_ind
    
    # Function to generate initial population of mutated sequences
    def generate_population(self, original_seq, pop_size):
        population = []
        for _ in range(pop_size):
            mutated_seq = random.choice([self.mutate_swap, self.mutate_reverse, self.mutate_shuffle])(original_seq)
            population.append(mutated_seq)
        return population

    # Main genetic algorithm loop
    def genetic_algorithm(self, original_seq, original_label, label_func, pop_size=100, generations=50, good_threshold=0.8):
        population = self.generate_population(original_seq, pop_size)
        good_examples, bad_examples = [], []

        for generation in range(generations):
            fitness_scores = []
            for candidate_seq in population:
                candidate_label = label_func(candidate_seq)  # Function to assign label based on the recommender system
                similarity, label_dist, self_ind = self.fitness(original_seq, candidate_seq, original_label, candidate_label)
                if self_ind != 0:
                    continue
                if similarity >= good_threshold and label_dist == 0:  # Good example
                    good_examples.append((candidate_seq, candidate_label))
                elif similarity >= good_threshold and label_dist == 1:  # Bad example
                    bad_examples.append((candidate_seq, candidate_label))

                fitness_scores.append((similarity, label_dist))

            # Select top sequences based on fitness (similarity and label match)
            population = [population[i] for i in np.argsort([-s for s, _ in fitness_scores])[:pop_size // 2]]

            # Apply mutations to generate the next population
            next_population = []
            for seq in population:
                next_population.append(random.choice([self.mutate_swap, self.mutate_reverse, self.mutate_shuffle])(seq))
            population = next_population

        return good_examples, bad_examples

    def generate(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        def label_func(seq):
            #NOTE: dummy black box model, use recommender
            return 1 if sum(seq) % 2 == 0 else 0

        original_label = label_func(x)

        return self.genetic_algorithm(original_seq=x, label_func=label_func, original_label=original_label, pop_size=10)

    


def generate(strategy: GenerationStrategy):
    src = np.random.randint(0,100, (20,))
    result = strategy.generate(src)
    print(result)



if __name__ == "__main__":
    generate(strategy=GeneticGenerationStrategy())
