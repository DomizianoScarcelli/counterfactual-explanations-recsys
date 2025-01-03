import math
from statistics import mean
from typing import Callable, List, Optional, Set

import numpy as np
import torch
from deap import tools
from torch import Tensor

from config import ConfigParams
from constants import MAX_LENGTH, cat2id
from generation.extended_ea_algorithms import eaSimpleBatched
from generation.mutations import ALL_MUTATIONS, Mutation
from generation.strategies.genetic import GeneticStrategy
from generation.utils import (
    _evaluate_categorized_generation,
    equal_ys,
    get_category_map,
    labels2cat,
)
from models.utils import pad_batch, topk, trim
from type_hints import CategorizedDataset
from utils_classes.distances import (
    edit_distance,
    intersection_weighted_ndcg,
    self_indicator,
)
from utils_classes.Split import Split


class TargetedGeneticStrategy(GeneticStrategy):
    def __init__(
        self,
        input_seq: Tensor,
        target: Set[int] | Set[str],
        model: Callable,
        alphabet: List[int],
        allowed_mutations: List[Mutation] = ALL_MUTATIONS,
        pop_size: int = ConfigParams.POP_SIZE,
        generations: int = ConfigParams.GENERATIONS,
        k: int = min(ConfigParams.TOPK),  # NOTE: if this is 1, the results are better.
        good_examples: bool = True,
        halloffame_ratio: float = 0.1,
        verbose: bool = True,
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
        )
        self.category_map = get_category_map()
        self.k = k
        if all(isinstance(t, str) for t in target):
            target = {cat2id[t] for t in target}  # type: ignore

        assert all(
            isinstance(t, int) for t in target
        ), f"Something went wrong, target must be a set of int, not {target}"

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
            y_primes = topk(
                logits=candidate_preds, dim=-1, k=self.k, indices=True
            )  # [num_seqs, k]

            # I get the category identifiers (list) of each candidate
            y_primes = [
                labels2cat(y_prime, encode=True) for y_prime in y_primes
            ]  # [num_seqs, k]

            topk_ys = topk(logits=self.gt, dim=-1, k=self.k, indices=True)  # shape [k]
            ys = labels2cat(topk_ys, encode=True)  # shape [k]

            target_ys = [self.target for _ in range(self.k)]  # [k]

            for n_i in range(candidate_seqs.size(0)):
                candidate_seq = trim(candidate_seqs[n_i])

                seq_dist = edit_distance(
                    self.input_seq, candidate_seq, normalized=True
                )  # [0,MAX_LENGTH] if not normalized, [0,1] if normalized
                cat_dist_from_gt = 1 - intersection_weighted_ndcg(ys, y_primes[n_i])
                cat_dist_from_target = 1 - intersection_weighted_ndcg(
                    target_ys, y_primes[n_i]
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

        # print(f"[DEBUG] Fitnesses: {list(round(x[0], 3) for x in sorted(fitnesses))[:20]}")
        return fitnesses

    def generate(self) -> CategorizedDataset:  # type: ignore
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
        """Removes the bad points from the good points and vice versa."""
        target_cats = [self.target for _ in range(self.k)]
        if self.good_examples:
            clean: CategorizedDataset = [
                (seq, cats) for seq, cats in examples if equal_ys(target_cats, cats)
            ]
            self.print(
                f"Removed {len(examples) - len(clean)} individuals from good (label was not equal to gt)"
            )
            return clean
        clean: CategorizedDataset = [
            (seq, cats) for seq, cats in examples if not equal_ys(target_cats, cats)
        ]
        self.print(
            f"Removed {len(examples) - len(clean)} individuals from bad (label was equal to gt)"
        )
        return clean

    def _postprocess(self, population: CategorizedDataset) -> CategorizedDataset:  # type: ignore
        clean_pop = self._clean(population)
        label_eval, seq_eval = self.evaluate_generation(clean_pop)

        # Remove any copy of the source point from the good or bad dataset.
        new_pop = [
            (ind, label)
            for ind, label in clean_pop
            if ind.tolist() != self.input_seq.tolist()
        ]
        if len(new_pop) < len(clean_pop):
            self.print(
                f"Source point was in the dataset {len(clean_pop) - len(new_pop)} times!, removing it"
            )
        clean_pop = new_pop

        # NOTE: I don't think this is needed
        # # If source point is not in good datset, add just one instance back
        # if (
        #     self.good_examples
        #     and len(
        #         [
        #             ind
        #             for ind, _ in clean_pop
        #             if ind.tolist() == source_point[0].tolist()
        #         ]
        #     )
        #     == 0
        # ):
        #     clean_pop.append(source_point)

        label_eval, seq_eval = self.evaluate_generation(clean_pop)
        self.print(
            f"[After clean] Good examples={self.good_examples} ({len(clean_pop)}) ratio of same_label is: {label_eval*100}%, avg distance: {seq_eval}"
        )
        return clean_pop

    def evaluate_generation(self, examples: CategorizedDataset):  # type: ignore
        target_cats = [self.target for _ in range(self.k)]

        # TODO: return also the average score in the evaluation
        return _evaluate_categorized_generation(
            input_seq=self.input_seq,
            dataset=examples,
            cats=target_cats,
        )
