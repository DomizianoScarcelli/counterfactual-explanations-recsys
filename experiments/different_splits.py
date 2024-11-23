import fire
from recbole.model.abstract_recommender import SequentialRecommender
from tqdm import tqdm

from config import DATASET, MODEL
from exceptions import (CounterfactualNotFound, DfaNotAccepting,
                        DfaNotRejecting, NoTargetStatesError)
from models.config_utils import generate_model, get_config
from performance_evaluation.alignment.utils import preprocess_interaction
from run import single_run
from utils import set_seed
from utils_classes.generators import DatasetGenerator
from utils_classes.Split import Split


def different_splits(use_cache: bool=False, num_counterfactuals: int = 10):
    """
    The experiment consists in evaluating the counterfactual generation
    algorithm on different splits of the source sequence. A spilt defines the
    parts of the source input that can be modified in order to produce a
    counterfactual. From this experiment we want to know the percentage of
    valid counterfactual generated for each split. In particular, we would like
    to see how the percentage changes when we force the model to edit only the
    first elements of the sequence; the middle elements; and the latest
    elements.
    """
    set_seed()
    # Create interaction and dataset genrations + oracle model
    config = get_config(dataset=DATASET, model=MODEL)
    oracle: SequentialRecommender = generate_model(config)
    datasets = DatasetGenerator(config=config, use_cache=use_cache, return_interaction=True)
    
    #TODO: log results

    # Define the set of splits to evaluate
    splits = {"start": Split(0, 10, None), #No executed part; mutable part is the 10 items, the rest is the fixed part
              "middle": Split(None, 10, None), # Executed part and fixed part are equal, meaning mutable part is 10 items in the center
              "end": Split(None, 10, 0)} # No fixed part, meaning mutable part are the 10 items at the end
    
    # Evaluation loop
    good, bad, not_valid = 0, 0, 0
    for i, (dataset, interaction) in enumerate(tqdm(datasets, desc="Different Splits Experiment...", total=num_counterfactuals)):
        if i == num_counterfactuals:
            print(f"Generated {num_counterfactuals}, exiting...")
            break

        source_sequence, source_gt = preprocess_interaction(interaction, oracle)
        for split_type, split in splits.items():
            split = split.parse_nan(source_sequence)
            try:
                aligned, _, _ = single_run(source_sequence=source_sequence, _dataset=dataset, split=split)
                aligned_gt = oracle(aligned).argmax(-1).item()
                if aligned_gt != source_gt:
                    bad += 1
                    print(f"Alignment for split: {split_type} is BAD!")
                else:
                    good += 1
                    print(f"Alignment for split: {split_type} is GOOD!")
            except (DfaNotAccepting, DfaNotRejecting, NoTargetStatesError, CounterfactualNotFound) as e:
                not_valid +=1
                print(f"Alignment for split: {split_type} is NOT VALID!, raised {type(e)}")
                continue


        print(f"i: {i} | Good: {good}, bad: {bad}, not_valid: {not_valid}")

if __name__ == "__main__":
    fire.Fire(different_splits)
            

             
