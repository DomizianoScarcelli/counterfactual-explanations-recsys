from utils import printd
import warnings
from itertools import product
from pathlib import Path

import pandas as pd
from torch import Tensor
from tqdm import tqdm

from config import ConfigParams
from generation.mutations import parse_mutations
from generation.strategies.genetic import GeneticStrategy
from generation.utils import _evaluate_generation, get_items
from models.config_utils import generate_model, get_config
from models.model_funcs import model_predict
from performance_evaluation.alignment.utils import log_run
from type_hints import Dataset
from utils_classes.generators import SequenceGenerator

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def update_config(current_config):
    default_config = ConfigParams.get_default_config()
    allowed_mutations, mutations_param, fitness_alpha, pop_size, halloffame_ratio = (
        current_config
    )
    default_config["evolution"]["allowed_mutations"] = allowed_mutations
    default_config["evolution"]["mutations"]["num_replaces"] = mutations_param
    default_config["evolution"]["mutations"]["num_additions"] = mutations_param
    default_config["evolution"]["mutations"]["num_deletions"] = mutations_param
    default_config["evolution"]["fitness_alpha"] = fitness_alpha
    default_config["evolution"]["pop_size"] = pop_size
    default_config["evolution"]["halloffame_ratio"] = halloffame_ratio
    ConfigParams.reload_from_dict(default_config)
    ConfigParams.print_config()


def evaluate_dataset(sequence: Tensor, examples: Dataset, label: int):
    expected_pop_size = ConfigParams.POP_SIZE
    pop_size = len(examples)
    pop_ratio = pop_size / expected_pop_size

    _, (norm_seq_dist, seq_dist) = _evaluate_generation(sequence, examples, label)

    return {
        "pop_ratio": pop_ratio,
        "normalized_seq_dist": norm_seq_dist,
        "seq_dist": seq_dist,
    }


def evaluate_config(sequence: Tensor, model, alphabet, seq_index: int):
    log_save_path = Path("results/genetic_evaluation.csv")
    prev_df = pd.DataFrame({})
    if log_save_path.exists():
        prev_df = pd.read_csv(log_save_path)

    allowed_mutations = parse_mutations(ConfigParams.ALLOWED_MUTATIONS)

    good_genetic_strategy = GeneticStrategy(
        input_seq=sequence,
        model=lambda x: model_predict(seq=x, model=model, prob=True),
        allowed_mutations=allowed_mutations,
        pop_size=ConfigParams.POP_SIZE,
        good_examples=True,
        generations=ConfigParams.GENERATIONS,
        halloffame_ratio=ConfigParams.HALLOFFAME_RATIO,
        alphabet=alphabet,
        verbose=False,
    )
    good_examples = good_genetic_strategy.generate()

    good_log = evaluate_dataset(
        sequence, good_examples, good_genetic_strategy.gt.argmax(-1).item()
    )
    good_log = {f"good_{key}": value for key, value in good_log.items()}

    bad_genetic_strategy = GeneticStrategy(
        input_seq=sequence,
        model=lambda x: model_predict(seq=x, model=model, prob=True),
        allowed_mutations=allowed_mutations,
        pop_size=ConfigParams.POP_SIZE,
        good_examples=False,
        generations=ConfigParams.GENERATIONS,
        halloffame_ratio=ConfigParams.HALLOFFAME_RATIO,
        alphabet=alphabet,
        verbose=False,
    )
    bad_examples = bad_genetic_strategy.generate()

    bad_log = evaluate_dataset(
        sequence, bad_examples, bad_genetic_strategy.gt.argmax(-1).item()
    )
    bad_log = {f"bad_{key}": value for key, value in bad_log.items()}
    log = {"seq_idx": seq_index, **good_log, **bad_log}
    prev_df = log_run(prev_df, log, log_save_path)


def main():
    # search space
    mutations_params = [1, 2, 4]
    allowed_mutations_list = [
        ["replace", "swap", "add", "delete", "shuffle", "reverse"],
        ["add", "delete"],
        ["replace", "swap"],
        ["replace", "swap", "add", "delete"],
    ]
    fitness_alphas = [0, 0.25, 0.5, 0.75]
    pop_sizes = [512, 1024, 2048]
    halloffame_ratios = [0, 0.2]

    # Create the Cartesian product
    permutations = list(
        product(
            allowed_mutations_list,
            mutations_params,
            fitness_alphas,
            pop_sizes,
            halloffame_ratios,
        )
    )

    printd(f"[Info] Creating model and sequence generator...")
    conf = get_config(ConfigParams.DATASET, ConfigParams.MODEL)
    sequences = SequenceGenerator(conf)
    model = generate_model(conf)
    alphabet = list(get_items())
    printd(f"[Info] Finished creating stuff, starting evaluation...")
    for current_config in tqdm(permutations, desc="Evaluating generation..."):

        steps = 1  # number of generations to perform for each configuration
        update_config(current_config)
        for sequence in sequences:
            sequence = sequence.squeeze()
            if sequences.index > steps:
                break
            evaluate_config(sequence, model, alphabet, sequences.index)
        sequences.reset()


if __name__ == "__main__":
    main()
