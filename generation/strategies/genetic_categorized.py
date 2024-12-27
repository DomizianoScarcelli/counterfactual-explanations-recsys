from typing import Optional
from utils_classes.Split import Split
from generation.utils import labels2cat
from utils_classes.distances import intersection_weighted_ndcg
from models.utils import topk
import math
from typing import Callable, List

import numpy as np
import torch
from deap import tools
from torch import Tensor

from config import ConfigParams
from constants import MAX_LENGTH
from generation.extended_ea_algorithms import eaSimpleBatched
from generation.mutations import ALL_MUTATIONS, Mutation
from generation.strategies.genetic import GeneticStrategy
from generation.utils import (
    _evaluate_categorized_generation,
    equal_ys,
    get_category_map,
)
from models.utils import pad_batch, trim
from type_hints import CategorizedDataset
from utils_classes.distances import edit_distance, self_indicator


class CategorizedGeneticStrategy(GeneticStrategy):
    """
    A specialized genetic strategy designed to optimize input sequences with a focus on maintaining or altering
    their categorical alignment, depending on the configuration.

    This class uses genetic algorithms to evolve sequences while incorporating a fitness evaluation based on
    sequence similarity and categorical distribution alignment.
    """

    def __init__(
        self,
        input_seq: Tensor,
        model: Callable,
        alphabet: List[int],
        allowed_mutations: List[Mutation] = ALL_MUTATIONS,
        pop_size: int = ConfigParams.POP_SIZE,
        generations: int = ConfigParams.GENERATIONS,
        k: int = ConfigParams.TOPK,
        good_examples: bool = True,
        halloffame_ratio: float = 0.1,
        verbose: bool = True,
        split: Optional[Split] = None,
    ):
        """
        Initializes the CategorizedGeneticStrategy.

        Args:
            input_seq (Tensor): The initial sequence to optimize.
            model (Callable): A model that predicts the fitness of sequences.
            alphabet (List[int]): The set of valid characters for sequence generation.
            allowed_mutations (List[Mutation], optional): Allowed mutations for sequence evolution.
                Defaults to ALL_MUTATIONS.
            pop_size (int, optional): Population size for the genetic algorithm. Defaults to ConfigParams.POP_SIZE.
            generations (int, optional): Number of generations to run. Defaults to ConfigParams.GENERATIONS.
            k (int, optional): Number of top predictions to consider for fitness evaluation. Defaults to ConfigParams.TOPK.
            good_examples (bool, optional): Whether to focus on examples matching the input category. Defaults to True.
            halloffame_ratio (float, optional): Proportion of the population kept in the hall of fame. Defaults to 0.1.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to True.
        """

        super().__init__(
            input_seq=input_seq,
            model=model,
            alphabet=alphabet,
            allowed_mutations=allowed_mutations,
            pop_size=pop_size,
            generations=generations,
            good_examples=good_examples,
            halloffame_ratio=halloffame_ratio,
            verbose=verbose,
            split=split,
        )
        self.category_map = get_category_map()
        self.k = k

    def evaluate_fitness_batch(self, individuals: List[List[int]]) -> List[float]:
        """
        Evaluates the fitness of a batch of individuals based on sequence similarity and categorical alignment.

        Args:
            individuals (List[List[int]]): A batch of sequences to evaluate.

        Returns:
            List[float]: A list of fitness scores for each sequence.
        """
        batch_size = 512
        num_batches = math.ceil(len(individuals) / batch_size)
        fitnesses = []
        ALPHA2 = 1 - ConfigParams.FITNESS_ALPHA
        for batch_i in range(num_batches):
            batch_individuals = individuals[
                batch_i * batch_size : (batch_i + 1) * batch_size
            ]
            candidate_seqs = pad_batch(batch_individuals, MAX_LENGTH)  # [num_seqs]

            candidate_preds = self.model(candidate_seqs)
            topk_y_primes = topk(logits=candidate_preds, dim=-1, k=self.k, indices=True)

            y_primes = [
                labels2cat(topk_y_prime, encode=True) for topk_y_prime in topk_y_primes
            ]  # [num_seqs, k]

            topk_ys = topk(logits=self.gt, dim=-1, k=self.k, indices=True).squeeze(
                0
            )  # shape [k]
            ys = labels2cat(topk_ys, encode=True)  # shape [k]

            for n_i in range(candidate_seqs.size(0)):
                candidate_seq = trim(candidate_seqs[n_i])

                # assert self.gt.shape == candidate_y.shape, f"Shape mismatch: {self.gt.shape} != {candidate_y.shape}"
                seq_dist = edit_distance(
                    self.input_seq, candidate_seq, normalized=True
                )  # [0,MAX_LENGTH] if not normalized, [0,1] if normalized
                cat_dist = 1 - intersection_weighted_ndcg(ys, y_primes[n_i])

                # print(f"Seq dist: {seq_dist}")
                # print(f"Cat dist: {cat_dist}")

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

        # print(f"[DEBUG] Fitnesses: {list(round(x[0], 3) for x in sorted(fitnesses))[:20]}")
        return fitnesses

    def generate(self) -> CategorizedDataset:  # type: ignore
        """
        Runs the genetic algorithm to produce a new set of optimized sequences.

        Returns:
            CategorizedDataset: A dataset of optimized sequences with their predicted categories.
        """
        population = self.toolbox.population()

        halloffame_size = int(np.round(self.pop_size * self.halloffame_ratio))
        halloffame = tools.HallOfFame(halloffame_size)

        population, _ = eaSimpleBatched(
            population,
            self.toolbox,
            cxpb=ConfigParams.CROSSOVER_PROB,
            mutpb=ConfigParams.MUT_PROB,
            ngen=self.generations,
            halloffame=halloffame if self.halloffame_ratio != 0 else None,
            verbose=False,
            pbar=self.verbose,
        )
        preds = self.model(pad_batch(population, MAX_LENGTH))
        preds = topk(logits=preds, dim=-1, k=self.k, indices=True)  # [pop_size, k]
        cats = [labels2cat(y, encode=True) for y in preds]  # [pop_size, k]

        # cats = labels2cat(preds, encode=True)
        new_population: CategorizedDataset = [
            (torch.tensor(ind), cat) for ind, cat in zip(population, cats)
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
        preds = self.model(pad_batch(augmented, MAX_LENGTH))
        preds = topk(logits=preds, dim=-1, k=self.k, indices=True)
        cats = [labels2cat(y, encode=True) for y in preds]  # [pop_size, k]

        for ind, cat in zip(augmented, cats):
            new_augmented.append((torch.tensor(ind), cat))

        label_eval, seq_eval = self.evaluate_generation(new_augmented)
        self.print(
            f"[Augmented] Good examples = {self.good_examples} [{len(new_augmented)}] ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}"
        )
        return self._postprocess(new_augmented)

    def _clean(self, examples: CategorizedDataset) -> CategorizedDataset:  # type: ignore
        """
        Cleans the population by removing undesired sequences based on category alignment.

        Args:
            examples (CategorizedDataset): The dataset to clean.

        Returns:
            CategorizedDataset: A cleaned dataset of sequences.
        """
        gt_preds = topk(logits=self.gt, dim=-1, k=self.k, indices=True).squeeze(0)
        gt_cats = labels2cat(gt_preds, encode=True)
        if self.good_examples:
            clean: CategorizedDataset = [
                (seq, cats) for seq, cats in examples if equal_ys(gt_cats, cats)
            ]
            self.print(
                f"Removed {len(examples) - len(clean)} individuals from good (label was not equal to gt)"
            )
            return clean
        clean: CategorizedDataset = [
            (seq, cats) for seq, cats in examples if not equal_ys(gt_cats, cats)
        ]
        self.print(
            f"Removed {len(examples) - len(clean)} individuals from bad (label was equal to gt)"
        )
        return clean

    def _postprocess(self, population: CategorizedDataset) -> CategorizedDataset:  # type: ignore
        """
        Post-processes the population by cleaning and optionally adding the source sequence.

        Args:
            population (CategorizedDataset): The dataset to process.

        Returns:
            CategorizedDataset: The processed dataset.
        """
        clean_pop = self._clean(population)
        label_eval, seq_eval = self.evaluate_generation(clean_pop)

        gt_preds = topk(logits=self.gt, dim=-1, k=self.k, indices=True).squeeze(0)
        gt_cats = labels2cat(gt_preds, encode=True)
        source_point = (self.input_seq, gt_cats)

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

    def evaluate_generation(self, examples: CategorizedDataset):  # type: ignore
        """
        Evaluates the generated dataset in terms of category matching and sequence similarity.

        Args:
            examples (CategorizedDataset): The dataset to evaluate.

        Returns:
            Tuple[float, float]: The percentage of sequences matching the input category and
            the average sequence distance.
        """
        gt_preds = topk(logits=self.gt, dim=-1, k=self.k, indices=True).squeeze(0)
        gt_cats = labels2cat(gt_preds, encode=True)
        # TODO: return also the average score in the evaluation
        return _evaluate_categorized_generation(
            input_seq=self.input_seq,
            dataset=examples,
            cats=gt_cats,
        )
