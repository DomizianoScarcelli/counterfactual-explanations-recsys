from typing import Optional
from utils_classes.Split import Split
import math
import random
from typing import Any, Callable, List

import numpy as np
import torch
from deap import base, creator, tools
from torch import Tensor

from config import ConfigParams
from constants import MAX_LENGTH, MIN_LENGTH, PADDING_CHAR
from generation.extended_ea_algorithms import (
    eaSimpleBatched,
    indexedCxTwoPoint,
    indexedSelTournament,
)
from generation.mutations import (
    ALL_MUTATIONS,
    AddMutation,
    DeleteMutation,
    Mutation,
    contains_mutation,
    remove_mutation,
)
from generation.strategies.abstract_strategy import GenerationStrategy
from generation.utils import _evaluate_generation, clone
from models.utils import pad_batch, trim
from type_hints import Dataset
from utils_classes.distances import (
    edit_distance,
    jensen_shannon_divergence,
    self_indicator,
)


class GeneticStrategy(GenerationStrategy):
    def __init__(
        self,
        input_seq: Tensor,
        model: Callable,
        alphabet: List[int],
        allowed_mutations: List[Mutation] = ALL_MUTATIONS,
        pop_size: int = ConfigParams.POP_SIZE,
        generations: int = ConfigParams.GENERATIONS,
        good_examples: bool = True,
        halloffame_ratio: float = 0.1,
        verbose: bool = True,
        split: Optional[Split] = None,
    ):
        super().__init__(
            input_seq=trim(input_seq),
            model=model,
            alphabet=alphabet,
            good_examples=good_examples,
            verbose=verbose,
        )
        self.pop_size = pop_size
        self.gt = self.model(input_seq.unsqueeze(0)).squeeze()
        self.generations = generations
        self.halloffame_ratio = halloffame_ratio
        self.allowed_mutations = allowed_mutations
        self.split = split
        if self.split:
            for mutation in self.allowed_mutations:
                mutation.set_split(self.split)
        # Define the evaluation function
        creator.create("fitness", base.Fitness, weights=(-1.0,))  # Minimize fitness
        creator.create("individual", list, fitness=creator.fitness)

        self.toolbox: Any = base.Toolbox()
        self.toolbox.register("feature_values", lambda x: x.tolist(), self.input_seq)
        self.toolbox.register(
            "individual",
            tools.initIterate,
            creator.individual,
            self.toolbox.feature_values,
        )
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,
            n=self.pop_size,
        )
        self.toolbox.register("clone", clone)

        self.toolbox.register("evaluate", self.evaluate_fitness_batch)
        self.toolbox.register("mate", indexedCxTwoPoint)  # Use two-point crossover
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register(
            "select", indexedSelTournament, tournsize=3
        )  # Tournament selection

    def print(self, s):
        if self.verbose:
            print(s)

    def mutate(self, seq: List[int], index: int):
        # Set seed according to the index in order to always choose a different mutation
        # TODO: remove for efficiency
        assert (
            PADDING_CHAR not in seq
        ), f"Seq must not contain padding char {PADDING_CHAR}: {seq}"
        mutations = self.allowed_mutations
        # If after NUM_ADDITIONS additions the seq is longer than the MAX_LENGTH, don't allow add mutations
        if len(seq) > MAX_LENGTH - ConfigParams.NUM_ADDITIONS and contains_mutation(
            AddMutation, mutations
        ):
            mutations = remove_mutation(AddMutation, mutations)
        # If after NUM_DELETIONS deletions the seq is shorter than the MIN_LENGTh, don't allow delete mutations
        if len(seq) < MIN_LENGTH + ConfigParams.NUM_DELETIONS and contains_mutation(
            DeleteMutation, mutations
        ):
            mutations = remove_mutation(DeleteMutation, mutations)
        mutation = random.choice(mutations)
        result = mutation(seq, self.alphabet, index)
        return result

    def evaluate_fitness_batch(self, individuals: List[List[int]]) -> List[float]:
        """
        Evaluates the fitness for each individual, feeding the individuals into the predictor in batches.
        """
        batch_size = 512
        num_batches = math.ceil(len(individuals) / batch_size)
        fitnesses = []
        ALPHA2 = 1 - ConfigParams.FITNESS_ALPHA
        for batch_i in range(num_batches):
            batch_individuals = individuals[
                batch_i * batch_size : (batch_i + 1) * batch_size
            ]
            candidate_seqs = pad_batch(batch_individuals, MAX_LENGTH)
            candidate_probs = self.model(candidate_seqs)

            # TODO: you can use this to remove the for loop. This still doesn't work
            # edit distance is not easily vectorizable
            # seq_dists = list(map(lambda seq: edit_distance(self.input_seq, trim(seq)), candidate_seqs))
            # if self.good_examples:
            #     label_dists = cosine_distance(self.gt, candidate_probs)
            # else:
            #     label_dists = 1 - cosine_distance(self.gt, candidate_probs)
            # # also self inds can be vectorized
            # self_inds = list(map(lambda seq: self_indicator(self.input_seq, trim(seq)), candidate_seqs))
            # costs = [ALPHA1 * seq_dist + ALPHA2 * label_dist + self_ind for (seq_dist, label_dist, self_ind) in zip(seq_dists, label_dists, self_inds)]
            # fitnesses.extend(costs)

            for i in range(candidate_seqs.size(0)):
                candidate_seq = trim(candidate_seqs[i])
                candidate_prob = candidate_probs[i]

                assert self.gt.shape == candidate_prob.shape
                seq_dist = edit_distance(
                    self.input_seq, candidate_seq, normalized=True
                )  # [0,MAX_LENGTH] if not normalized, [0,1] if normalized
                label_dist = jensen_shannon_divergence(candidate_prob, self.gt)  # [0,1]
                self_ind = self_indicator(
                    self.input_seq, candidate_seq
                )  # 0 if different, inf if equal
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
                cost = (
                    ConfigParams.FITNESS_ALPHA * seq_dist
                    + ALPHA2 * label_dist
                    + self_ind,
                )
                fitnesses.append(cost)

        # print(f"[DEBUG] Fitnesses: {list(sorted(fitnesses))}")
        return fitnesses

    def generate(self) -> Dataset:
        population = self.toolbox.population()

        halloffame_size = int(np.round(self.pop_size * self.halloffame_ratio))
        halloffame = tools.HallOfFame(halloffame_size)

        population, _ = eaSimpleBatched(
            population,
            self.toolbox,
            cxpb=0.7,
            mutpb=0.5,
            ngen=self.generations,
            halloffame=halloffame if self.halloffame_ratio != 0 else None,
            verbose=False,
            pbar=self.verbose,
        )
        preds = self.model(pad_batch(population, MAX_LENGTH)).argmax(-1)
        new_population = [
            (torch.tensor(x), preds[i].item()) for (i, x) in enumerate(population)
        ]
        label_eval, seq_eval = self.evaluate_generation(new_population)
        self.print(
            f"[Original] Good examples = {self.good_examples} [{len(new_population)}] ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}"
        )
        if not self.good_examples or self.halloffame_ratio == 0:
            # Augment only good examples, which are the rarest
            return self._postprocess(new_population)

        augmented = self._augment(population, halloffame)
        new_augmented = []
        preds = self.model(pad_batch(augmented, MAX_LENGTH)).argmax(-1)
        for i, x in enumerate(augmented):
            new_augmented.append((torch.tensor(x), preds[i].item()))
        label_eval, seq_eval = self.evaluate_generation(new_augmented)
        self.print(
            f"[Augmented] Good examples = {self.good_examples} [{len(new_augmented)}] ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}"
        )
        return self._postprocess(new_augmented)

    def _augment(self, population, halloffame):
        fitness_values = [
            p.fitness.wvalues[0]
            for p in population
            if p.fitness.wvalues[0] != float("-inf")
        ]
        fitness_values = sorted(fitness_values)
        fitness_diff = [
            fitness_values[i + 1] - fitness_values[i]
            for i in range(0, len(fitness_values) - 1)
        ]

        index = np.max(
            np.argwhere(fitness_diff == np.amax(fitness_diff)).flatten().tolist()
        )
        fitness_value_thr = fitness_values[index]

        oversample = list()

        for p in population:
            if p.fitness.wvalues[0] > fitness_value_thr:
                oversample.append(list(p))

        for h in halloffame:
            if h.fitness.wvalues[0] > fitness_value_thr:
                oversample.append(list(h))

        return oversample

    def _clean(self, examples: Dataset) -> Dataset:
        label = self.gt.argmax(-1).item()
        if self.good_examples:
            clean: Dataset = [ex for ex in examples if ex[1] == label]
            self.print(
                f"Removed {len(examples) - len(clean)} individuals from good (label was not equal to gt)"
            )
            return clean
        clean: Dataset = [ex for ex in examples if ex[1] != label]
        self.print(
            f"Removed {len(examples) - len(clean)} individuals from bad (label was equal to gt)"
        )
        return clean

    def _postprocess(self, population: Dataset) -> Dataset:
        clean_pop = self._clean(population)
        label_eval, seq_eval = self.evaluate_generation(clean_pop)

        source_point = (self.input_seq, self.gt.argmax(-1).item())

        # Remove any copy of the source point from the good or bad dataset.
        new_pop = [
            (ind, label)
            for ind, label in clean_pop
            if ind.tolist() != source_point[0].tolist()
        ]
        if len(new_pop) < len(clean_pop):
            self.print(
                f"Source point was in the dataset {len(clean_pop) - len(new_pop)} times!, removing it"
            )
        clean_pop = new_pop

        # If source point is not in good datset, add just one instance back
        if (
            self.good_examples
            and len(
                [
                    ind
                    for ind, _ in clean_pop
                    if ind.tolist() == source_point[0].tolist()
                ]
            )
            == 0
        ):
            clean_pop.append(source_point)

        label_eval, seq_eval = self.evaluate_generation(clean_pop)
        self.print(
            f"[After clean] Good examples={self.good_examples} ({len(clean_pop)}) ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}"
        )
        return clean_pop

    def evaluate_generation(self, examples: Dataset):
        return _evaluate_generation(self.input_seq, examples, self.gt.argmax(-1).item())
