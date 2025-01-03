from performance_evaluation.alignment.evaluate import _log_error, evaluate_targeted
import json
import warnings
from typing import Generator, List, Optional, Dict
from tqdm import tqdm

from alignment.actions import print_action
from alignment.alignment import trace_disalignment
from alignment.utils import postprocess_alignment
from automata_learning.learning import learning_pipeline
from config import ConfigParams
from constants import MAX_LENGTH, cat2id
from exceptions import (
    CounterfactualNotFound,
    DfaNotAccepting,
    DfaNotRejecting,
    EmptyDatasetError,
    NoTargetStatesError,
    SplitNotCoherent,
)
from generation.dataset.utils import interaction_to_tensor
from generation.utils import equal_ys, labels2cat
from models.utils import pad, topk, trim
from performance_evaluation.alignment.utils import preprocess_interaction
from type_hints import GoodBadDataset, RecDataset, RecModel, SplitTuple, CategorySet
from utils import TimedFunction, seq_tostr
from utils_classes.distances import edit_distance
from utils_classes.generators import (
    DatasetGenerator,
    InteractionGenerator,
    TimedGenerator,
)
from utils_classes.Split import Split

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

timed_learning_pipeline = TimedFunction(learning_pipeline)
timed_trace_disalignment = TimedFunction(trace_disalignment)

error_messages = {
    DfaNotAccepting: "DfaNotAccepting",
    DfaNotRejecting: "DfaNotRejecting",
    NoTargetStatesError: "NoTargetStatesError",
    CounterfactualNotFound: "CounterfactualNotFound",
    SplitNotCoherent: "SplitNotCoherent",
    EmptyDatasetError: "EmptyDatasetError",
}



