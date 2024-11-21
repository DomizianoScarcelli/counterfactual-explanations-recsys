import os
from typing import Generator, Optional

import fire
import pandas as pd
import torch
from pandas import DataFrame
from recbole.model.abstract_recommender import SequentialRecommender
from tqdm import tqdm

from alignment.actions import print_action
from config import DATASET, GENERATIONS, HALLOFFAME_RATIO, MODEL, POP_SIZE
from constants import MAX_LENGTH
from exceptions import (CounterfactualNotFound, DfaNotAccepting,
                        DfaNotRejecting, NoTargetStatesError)
from genetic.dataset.generate import dataset_generator, interaction_generator
from models.config_utils import generate_model, get_config
from models.utils import pad, trim
from performance_evaluation.alignment.utils import (evaluate_stats, get_split,
                                                    is_already_evaluated,
                                                    log_run,
                                                    preprocess_interaction)
from run import single_run, timed_learning_pipeline, timed_trace_disalignment
from utils import set_seed


def different_splits():
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
    # Create interaction and dataset genrations + oracle model
    config = get_config(dataset=DATASET, model=MODEL)
    oracle: SequentialRecommender = generate_model(config)
    interactions = interaction_generator(config)
    datasets = dataset_generator(config=config, use_cache=use_cache)
    
    # Define the set of splits to evaluate
    abstract_splits = { (None, 10, None), (None, None, 10), (10, None, None) }


    pass
