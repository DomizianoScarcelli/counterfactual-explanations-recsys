import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
import random
import itertools
from tqdm import tqdm

# A function to calculate similarity between two sequences (e.g., edit distance)
def sequence_similarity(seq1, seq2):
    return np.sum(np.array(seq1) == np.array(seq2)) / len(seq1)  # Fraction of matching elements

# A function to calculate label distance (0 if same label, 1 if different)
def label_distance(label1, label2):
    return 0 if label1 == label2 else 1

def cosine_similarity(prob1, prob2):
    #TODO: add cosine similarity between original sequence probability
    #prediction, and new sequence prob prediction
    pass

def self_indicator(seq1, seq2):
    return float("inf") if (seq1 == seq2).all() else 0

def dummy_model(seq):
    #NOTE: dummy black box model
    return 1 if sum(seq) % 2 == 0 else 0

class GenerationStrategy(ABC):
    @abstractmethod
    def generate(self, x: np.ndarray,  model: Callable[[np.ndarray], int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pass

class RandomPickGenerationStrategy(GenerationStrategy):
    def generate(self, x: np.ndarray,  model: Callable[[np.ndarray], int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pass
    
class GeneticGenerationStrategy(GenerationStrategy):
    
    # Crossover: Two-point crossover between two sequences
    def two_point_crossover(self, seq1, seq2):
        # Ensure the sequences are the same length
        assert len(seq1) == len(seq2), "Sequences must be of equal length for crossover."

        # Copy sequences to avoid modifying the originals
        child1, child2 = seq1.copy(), seq2.copy()

        # Select two random crossover points
        i, j = sorted(random.sample(range(len(seq1)), 2))

        # Swap the segments between the two points
        child1[i:j+1], child2[i:j+1] = seq2[i:j+1], seq1[i:j+1]

        return child1, child2

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
        seq = sequence.copy()
        i, j = sorted(random.sample(range(len(seq)), 2))  # Select two random indices and sort them
        subseq = seq[i:j+1]  # Extract the subsequence between these indices
        random.shuffle(subseq)  # Shuffle the subsequence
        seq[i:j+1] = subseq  # Place the shuffled subsequence back into the original sequence
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
    def genetic_algorithm(self, original_seq, original_label, label_func, pop_size=100, generations=50, good_threshold=0.8, good_examples=True, pc: float=0.5, pm: float=0.5):
        population = self.generate_population(original_seq, pop_size)
        examples = []

        for generation in tqdm(range(generations), "Genetic algorithm..."):
            fitness_scores = []
            
            for candidate_seq in population:
                candidate_label = label_func(candidate_seq)  # Function to assign label based on the recommender system
                similarity, label_dist, self_ind = self.fitness(original_seq, candidate_seq, original_label, candidate_label)
                
                # Skip identical sequences (self-indicator)
                if self_ind != 0:
                    continue

                # Add to examples based on good_examples flag
                if good_examples:
                    if similarity >= good_threshold and label_dist == 0:
                        examples.append((candidate_seq, candidate_label))  # Good examples: similar & same label
                else:
                    if similarity >= good_threshold and label_dist == 1:
                        examples.append((candidate_seq, candidate_label))  # Bad examples: similar & different label

                fitness_scores.append((similarity, label_dist))

            # Select top sequences based on fitness (similarity and label match)
            population = [population[i] for i in np.argsort([-s for s, _ in fitness_scores])[:pop_size // 2]]

            # Crossover: generate new population from pairs of parents
            next_population = []
            to_be_removed = []
            while len(next_population) < pop_size * pc:
                #TODO: sample without repetition to avoid getting the same parents
                parent1, parent2 = random.sample(population, 2)
                # Perform two-point crossover to generate two children
                child1, child2 = self.two_point_crossover(parent1, parent2)
                next_population.extend([child1, child2])
                to_be_removed.extend([parent1.tolist(), parent2.tolist()])
            
            population = [pop for pop in population if pop.tolist() not in to_be_removed] + next_population
            # Apply mutations to the new population
            for i in range(len(population)):
                if random.random() < pm:
                    continue
                population[i] = random.choice([self.mutate_swap, self.mutate_reverse, self.mutate_shuffle])(population[i])

        return examples 

    def generate(self, x: np.ndarray, model: Callable[[np.ndarray], int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        original_label = model(x)

        good_examples = self.genetic_algorithm(original_seq=x, label_func=model, original_label=original_label, good_examples=True)
        bad_examples = self.genetic_algorithm(original_seq=x, label_func=model, original_label=original_label, good_examples=False)

        return good_examples, bad_examples

    


def generate(strategy: GenerationStrategy):
    src = np.random.randint(0,100, (20,))
    result = strategy.generate(src, dummy_model)
    print(result)



if __name__ == "__main__":
    generate(strategy=GeneticGenerationStrategy())
