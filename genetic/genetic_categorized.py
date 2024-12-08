from utils_classes.distances import jaccard_sim
from genetic.utils import label2cat
from statistics import mean
import math
from typing import Callable, List

import numpy as np
import torch
from deap import tools
from torch import Tensor

from config import ConfigParams
from constants import MAX_LENGTH
from genetic.extended_ea_algorithms import (
    eaSimpleBatched,
)
from genetic.genetic import GeneticStrategy
from genetic.mutations import (
    ALL_MUTATIONS,
    Mutation,
)
from genetic.utils import _evaluate_generation, get_category_map, label2cat
from models.utils import pad_batch, trim
from type_hints import Dataset
from utils import set_seed
from utils_classes.distances import (
    edit_distance,
    self_indicator,
)


class CategorizedGeneticStrategy(GeneticStrategy):
    def __init__(
        self,
        input_seq: Tensor,
        predictor: Callable,
        alphabet: List[int],
        allowed_mutations: List[Mutation] = ALL_MUTATIONS,
        pop_size: int = ConfigParams.POP_SIZE,
        generations: int = ConfigParams.GENERATIONS,
        good_examples: bool = True,
        halloffame_ratio: float = 0.1,
        verbose: bool = True,
    ):

        super().__init__(
            input_seq=input_seq,
            model=predictor,
            alphabet=alphabet,
            allowed_mutations=allowed_mutations,
            pop_size=pop_size,
            generations=generations,
            good_examples=good_examples,
            halloffame_ratio=halloffame_ratio,
            verbose=verbose,
        )
        self.category_map = get_category_map()

    def evaluate_fitness_batch(self, individuals: List[List[int]]) -> List[float]:
        """
        Evaluates the fitness for each individual, feeding the individuals into the predictor in batches.
        """
        set_seed()
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

            gt_cat = set(label2cat(self.gt.argmax(-1).item(), encode=True))

            for i in range(candidate_seqs.size(0)):
                candidate_seq = trim(candidate_seqs[i])
                candidate_ys = candidate_probs[i].argmax(-1)
                candidate_cats: List[List[int]] = [
                    label2cat(y.item(), encode=True) for y in candidate_ys
                ]

                assert self.gt.shape == candidate_ys.shape
                seq_dist = edit_distance(
                    self.input_seq, candidate_seq, normalized=True
                )  # [0,MAX_LENGTH] if not normalized, [0,1] if normalized
                sims = [
                    jaccard_sim(a=gt_cat, b=set(candidate_cat))
                    for candidate_cat in candidate_cats
                ]
                cat_dist = mean(1 - sim for sim in sims)

                print(f"[DEBUG] sims are: {sims}\n")
                print(f"[DEBUG] cat dist is: {cat_dist}\n")

                self_ind = self_indicator(
                    self.input_seq, candidate_seq
                )  # 0 if different, inf if equal
                if not self.good_examples:
                    # label_dist = 0 if label_dist == float("inf") else float("inf")
                    cat_dist = 1 - cat_dist
                cost = (
                    ConfigParams.FITNESS_ALPHA * seq_dist
                    + ALPHA2 * cat_dist
                    + self_ind,
                )
                fitnesses.append(cost)

        # print(f"[DEBUG] Fitnesses: {list(sorted(fitnesses))}")
        return fitnesses

    def generate(self) -> Dataset:
        set_seed()
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
        # TODO: convert preds to categories
        new_population = [
            (torch.tensor(x), preds[i].item()) for (i, x) in enumerate(population)
        ]
        label_eval, seq_eval = self.evaluate_generation(new_population)
        self.print(
            f"[Original] Good examples = {self.good_examples} [{len(new_population)}] ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}"
        )
        if not self.good_examples or self.halloffame_ratio == 0:
            # Augment only good examples, which are the rarest
            return new_population

        augmented = self._augment(population, halloffame)
        new_augmented = []
        preds = self.model(pad_batch(augmented, MAX_LENGTH)).argmax(-1)
        for i, x in enumerate(augmented):
            new_augmented.append((torch.tensor(x), preds[i].item()))
        label_eval, seq_eval = self.evaluate_generation(new_augmented)
        self.print(
            f"[Augmented] Good examples = {self.good_examples} [{len(new_augmented)}] ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}"
        )
        return new_augmented

    def _clean(self, examples: Dataset) -> Dataset:
        label = self.gt.argmax(-1).item()
        if self.good_examples:
            clean = [ex for ex in examples if ex[1] == label]
            self.print(
                f"Removed {len(examples) - len(clean)} individuals from good (label was not equal to gt)"
            )
            return clean
        clean = [ex for ex in examples if ex[1] != label]
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

    def evaluate_generation(self, examples):
        return _evaluate_generation(self.input_seq, examples, self.gt.argmax(-1).item())
