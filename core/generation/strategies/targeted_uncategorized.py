import math
from statistics import mean
from typing import Callable, List, Optional

import numpy as np
import torch
from deap import tools
from torch import Tensor

from config.config import ConfigParams
from config.constants import MAX_LENGTH
from core.generation.extended_ea_algorithms import eaSimpleBatched
from core.generation.mutations import ALL_MUTATIONS, Mutation
from core.generation.strategies.untargeted_uncategorized import GeneticStrategy
from core.generation.utils import (_evaluate_categorized_generation, equal_ys,
                                   get_category_map)
from core.models.utils import pad_batch, topk, trim
from type_hints import CategorizedDataset, Dataset
from utils.distances import (edit_distance, intersection_weighted_ndcg, ndcg,
                             self_indicator)
from utils.Split import Split


class TargetedUncategorizedGeneticStrategy(GeneticStrategy):
    def __init__(
        self,
        input_seq: Tensor,
        target: int,
        model: Callable,
        alphabet: List[int],
        allowed_mutations: List[Mutation] = ALL_MUTATIONS,
        pop_size: int = ConfigParams.POP_SIZE,
        generations: int = ConfigParams.GENERATIONS,
        k: int = ConfigParams.GENETIC_TOPK,
        good_examples: bool = True,
        halloffame_ratio: float = 0.1,
        verbose: bool = ConfigParams.DEBUG > 0,
        split: Optional[Split] = None,
    ):

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
            k=k,
        )
        self.category_map = get_category_map()
        self.target = target

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
            # Since the individuals is a list of lists, I need a tensor of equal-length sequences, so i pad them.
            candidate_seqs = pad_batch(batch_individuals, MAX_LENGTH)  # [num_seqs]
            candidate_preds = self.model(candidate_seqs)  # [num_seqs, num_items]
            y_primes = [
                [{x.item()}]
                for x in topk(logits=candidate_preds, dim=-1, k=self.k, indices=True)
            ]  # [num_seqs, k]

            topk_ys = [
                {x.item()} for x in topk(logits=self.gt, dim=-1, k=self.k, indices=True)
            ]  # shape [k]
            ys = topk_ys

            target_ys = [{self.target} for _ in range(self.k)]  # [k]

            for n_i in range(candidate_seqs.size(0)):
                candidate_seq = trim(candidate_seqs[n_i])

                seq_dist = edit_distance(
                    self.input_seq, candidate_seq, normalized=True
                )  # [0,MAX_LENGTH] if not normalized, [0,1] if normalized
                cat_dist_from_gt = 1 - intersection_weighted_ndcg(
                    ys, y_primes[n_i], perfect_score=1
                )
                cat_dist_from_target = 1 - intersection_weighted_ndcg(
                    target_ys, y_primes[n_i], perfect_score=1
                )
                cat_dist = cat_dist_from_target

                # 0 if different, inf if equal
                assert (
                    self.input_seq.dim() == 1
                ), f"input seq wrong shape: {self.input_seq.shape}"
                assert (
                    candidate_seq.dim() == 1
                ), f"candidate seq wrong shape: {candidate_seq.shape}"
                self_ind = self_indicator(self.input_seq, candidate_seq)
                if not self.good_examples:
                    cat_dist = mean([(1 - cat_dist), (1 - cat_dist_from_gt)])

                cost = (
                    ConfigParams.FITNESS_ALPHA * seq_dist
                    + ALPHA2 * cat_dist
                    + self_ind,
                )
                fitnesses.append(cost)

        return fitnesses

    def generate(self) -> Dataset:  # type: ignore
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
            split=self.split,
        )
        preds = self.model(pad_batch(population, MAX_LENGTH))
        preds = [
            [{x.item()}] for x in topk(logits=preds, dim=-1, k=self.k, indices=True)
        ]  # [pop_size, k]
        # cats = labels2cat(preds, encode=True)
        new_population: Dataset = [
            (torch.tensor(x), preds[i]) for (i, x) in enumerate(population)
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
        preds = [[{x.item()}] for x in topk(logits=preds, dim=-1, k=self.k, indices=True)]

        for i, x in enumerate(augmented):
            new_augmented.append((torch.tensor(x), preds[i]))

        label_eval, seq_eval = self.evaluate_generation(new_augmented)
        self.print(
            f"[Augmented] Good examples = {self.good_examples} [{len(new_augmented)}] ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}"
        )
        return self._postprocess(new_augmented)

    def _clean(self, examples: CategorizedDataset) -> CategorizedDataset:  # type: ignore
        """Removes the bad points from the good points and vice versa."""
        target = [{self.target} for _ in range(self.k)]
        if self.good_examples:
            clean: CategorizedDataset = [
                (seq, label) for seq, label in examples if equal_ys(target, label)
            ]
            self.print(
                f"Removed {len(examples) - len(clean)} individuals from good (label was not equal to gt)"
            )
            return clean
        clean: CategorizedDataset = [
            (seq, label) for seq, label in examples if not equal_ys(target, label)
        ]
        self.print(
            f"Removed {len(examples) - len(clean)} individuals from bad (label was equal to gt)"
        )
        return clean

    def evaluate_generation(self, examples: Dataset):  # type: ignore
        target = [{self.target} for _ in range(self.k)]

        # TODO: return also the average score in the evaluation
        return _evaluate_categorized_generation(
            input_seq=self.input_seq,
            dataset=examples,
            cats=target,
        )
