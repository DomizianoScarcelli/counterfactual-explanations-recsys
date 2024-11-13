import json
import warnings
from statistics import mean
from typing import Dict, Generator, List, Optional, Tuple, Union

from recbole.model.abstract_recommender import SequentialRecommender
from tqdm import tqdm

from config import DATASET, GENERATIONS, MODEL, POP_SIZE
from genetic.dataset.generate import interaction_generator
from genetic.dataset.utils import get_sequence_from_interaction
from genetic.genetic import GeneticGenerationStrategy
from genetic.mutations import (AddMutation, DeleteMutation, ReplaceMutation,
                               ReverseMutation, ShuffleMutation, SwapMutation)
from genetic.utils import NumItems
from models.config_utils import generate_model, get_config
from models.model_funcs import model_predict
from utils import set_seed

set_seed()
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def evaluation_step(sequence, model: SequentialRecommender):
    all_results = []
    search_mutations = [
            [SwapMutation(), ReplaceMutation()],
            [ReverseMutation(), ShuffleMutation()],
            [AddMutation(), DeleteMutation()],
            [SwapMutation(), ReplaceMutation(), AddMutation(), DeleteMutation()],
            [SwapMutation(), ReplaceMutation(), ReverseMutation(), ShuffleMutation()],
            [SwapMutation(), ReplaceMutation(), AddMutation(), DeleteMutation(), ReverseMutation(), ShuffleMutation()]
            ]
    for allowed_mutations in tqdm(search_mutations, desc="Evalutaion step..."):
        good_genetic_strategy = GeneticGenerationStrategy(input_seq=sequence,
                                                          allowed_mutations=allowed_mutations,
                                                          predictor=lambda x: model_predict(seq=x,
                                                                                            model=model,
                                                                                            prob=True),
                                                          pop_size=POP_SIZE,
                                                          good_examples=True,
                                                          generations=GENERATIONS,
                                                          verbose=False,
                                                          alphabet = set(range(NumItems.ML_1M.value)))
        good_examples = good_genetic_strategy.generate()
        good_same_label_perc, good_avg_distance = good_genetic_strategy.evaluate_generation(good_examples)
        len_good_examples = len(good_examples)
        good_examples = good_genetic_strategy.postprocess(good_examples)
        len_good_examples_post = len(good_examples)
        _, good_avg_distance_post = good_genetic_strategy.evaluate_generation(good_examples)

        bad_genetic_strategy = GeneticGenerationStrategy(input_seq=sequence,
                                                         allowed_mutations=allowed_mutations,
                                                         predictor=lambda x: model_predict(seq=x,
                                                                                           model=model,
                                                                                           prob=True),
                                                         pop_size=POP_SIZE,
                                                         good_examples=False,
                                                         generations=GENERATIONS,
                                                         verbose=False,
                                                         alphabet = set(range(NumItems.ML_1M.value)))
        bad_examples = bad_genetic_strategy.generate()
        bad_same_label_perc, bad_avg_distance = bad_genetic_strategy.evaluate_generation(bad_examples)
        len_bad_examples = len(bad_examples)
        bad_examples = bad_genetic_strategy.postprocess(bad_examples)
        len_bad_examples_post = len(bad_examples)
        _, bad_avg_distance_post = bad_genetic_strategy.evaluate_generation(bad_examples)

        results = {"mutations_allowed": [a.name for a in allowed_mutations], 
                   "generations": GENERATIONS,
                   "pop_size": POP_SIZE,
                   "good_stats": {"same_label_perc_pre": good_same_label_perc*100,
                                  "avg_distance_pre": good_avg_distance,
                                  "avg_distance_post": good_avg_distance_post,
                                  "len_population_pre": len_good_examples,
                                  "len_population_post": len_good_examples_post},

                   "bad_stats": {"same_label_perc_pre": bad_same_label_perc*100,
                                 "avg_distance_pre": bad_avg_distance,
                                 "avg_distance_post": bad_avg_distance_post,
                                 "len_population_pre": len_bad_examples,
                                 "len_population_post": len_bad_examples_post}}
        all_results.append(results)

    return all_results

def get_stats(results: Dict):
    """
    """
    mutation_stats = {} 
    for interaction_run in results.values():
        for mutation_run in interaction_run[0]: #TODO: next time the log is generated, with the extend, remove this [0]
            good_avg_distance_post = mutation_run["good_stats"]["avg_distance_post"]
            good_len_population_post = mutation_run["good_stats"]["len_population_post"]
            bad_avg_distance_post = mutation_run["bad_stats"]["avg_distance_post"]
            bad_len_population_post = mutation_run["bad_stats"]["len_population_post"]

            allowed_mutations = ", ".join(mutation_run["mutations_allowed"])
            if allowed_mutations not in mutation_stats:
                mutation_stats[allowed_mutations] = {"good_stats": {"avg_distance_post": [good_avg_distance_post], "len_population_post": [good_len_population_post]},
                                                     "bad_stats": {"avg_distance_post": [bad_avg_distance_post], "len_population_post": [bad_len_population_post]} }
            else:
                mutation_stats[allowed_mutations]["good_stats"]["avg_distance_post"].append(good_avg_distance_post)
                mutation_stats[allowed_mutations]["good_stats"]["len_population_post"].append(good_len_population_post)
                mutation_stats[allowed_mutations]["bad_stats"]["avg_distance_post"].append(bad_avg_distance_post)
                mutation_stats[allowed_mutations]["bad_stats"]["len_population_post"].append(bad_len_population_post)

    for mutations_comb in mutation_stats.values():
        mutations_comb["good_stats"]["avg_distance_post"] = mean(mutations_comb["good_stats"]["avg_distance_post"])
        mutations_comb["good_stats"]["len_population_post"] = mean(mutations_comb["good_stats"]["len_population_post"])
        mutations_comb["bad_stats"]["avg_distance_post"] = mean(mutations_comb["bad_stats"]["avg_distance_post"])
        mutations_comb["bad_stats"]["len_population_post"] = mean(mutations_comb["bad_stats"]["len_population_post"])

    with open("deap_generation_stats.json", "w") as f:
        json.dump(mutation_stats, f)


def evaluate_deap(start_from: Optional[str] = None, num_iterations: Optional[int] = 100):
    config = get_config(model=MODEL, dataset=DATASET)
    interactions = interaction_generator(config)
    model = generate_model(config)
    results = {}
    if start_from:
        with open(start_from, "r") as f:
            results = json.load(f)
    for idx, interaction in enumerate(interactions):
        if num_iterations and idx == num_iterations:
            print(f"Evaluated {num_iterations} interactions, ending evaluation.")
            break
        print(f"Evaluating interaction {idx}")
        if idx not in results:
            results[idx] = []
        else:
            print(f"Interaction {idx} already in results, skipping...")
            continue

        sequence = get_sequence_from_interaction(interaction).squeeze(0)
        curr_results = evaluation_step(sequence, model)
        results[idx].extend(curr_results)
        with open("deap_generator_log.json", "w") as f:
            json.dump(results, f)
            print("Results saved!")


if __name__ == "__main__":
    start_from = "deap_generator_log.json"
    # evaluate_deap(start_from=start_from, num_iterations=10)
    with open(start_from, "r") as f:
        results = json.load(f)
    get_stats(results)
