import pytest
import torch
from recbole.model.abstract_recommender import SequentialRecommender

from alignment.alignment import (augment_constraint_automata,
                                 augment_trace_automata)
from automata_learning.learning import (generate_automata_from_dataset,
                                        generate_single_accepting_sequence_dfa)
from config import ConfigParams
from genetic.dataset.utils import load_dataset
from genetic.utils import Items
from models.config_utils import generate_model, get_config
from utils_classes.generators import InteractionGenerator, SequenceGenerator


# By marking a class with @pytest.mark.incremental, if a test fails, all the other ones in the class are skipped
def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item

def pytest_runtest_setup(item):
    previousfailed = getattr(item.parent, "_previousfailed", None)
    if previousfailed is not None:
        pytest.xfail("previous test failed (%s)" % previousfailed.name)


@pytest.fixture(scope="module")
def mock_dataset():
    """
    A tiny and controllable set of good and bad points in order to visualize
    the learned automata and do further debug

    Assume the unviverse of all possible point (whic may not be contained in
                                                the dataset) is [1,2,3,4,5,6].
    This is equivalent to the universe of items in the recommender system real
    example
    """
    gp =[(torch.tensor([1,3,2]),True),
         (torch.tensor([1,2,3]),True),
         (torch.tensor([2,3,1,5]),True),
         (torch.tensor([5,4,1]),True)]

    bp =[(torch.tensor([1,2,4]),False),
         (torch.tensor([1,3,2,5]),False),
         (torch.tensor([1,2,5]),False), 
         (torch.tensor([5,3,4,1]),False), 
         (torch.tensor([2,3,5,1]),False)]
    return (gp, bp)

@pytest.fixture(scope="module")
def mock_automata(mock_dataset):
    """
    A tiny automata learned on the mock tiny dataset. This has no use beyond
    debugging and testing
    """
    good_points, bad_points = mock_dataset
    return generate_automata_from_dataset((good_points, bad_points), load_if_exists=False,  save_path="mock_automata.pickle")

# TODO: see how it's possible to combine mock and real fixtures/tests in order
# to not write a lot of repeated code
# (https://docs.pytest.org/en/stable/how-to/fixtures.html#factories-as-fixtures)
@pytest.fixture(scope="module")
def mock_original_trace(mock_dataset):
    gp, _ = mock_dataset
    return gp[0][0].tolist()

@pytest.fixture(scope="module")
def mock_bad_trace(mock_dataset):
    _, bp = mock_dataset
    return bp[3][0].tolist()

@pytest.fixture(scope="module")
def mock_edited_trace(mock_dataset):
    gp, _ = mock_dataset
    original_trace = gp[0][0].tolist()
    
    # Since we will be doing del_at, add_ot, we can only consider another good
    # trace where ot doesn't appear in the original trace
    trace_chars = set(original_trace)
    for another_trace, _ in gp:
        if len(set(another_trace) & trace_chars) == 0:
            break
    # [2720, 365, 1634, 1229, 140, 3298, 1664, 160, 1534, 1233, 618, 267, 2490, 2492, 2483, 89, 273, 665, 352, 222, 2265, 2612, 429, 213, 2827, 532, 1002, 202, 821, 1615, 1284, 830, 176, 1116, 2626, 23, 415, 1988, 694, 133, 1536, 510, 290, 152, 204, 1770, 1273, 289, 462, 165]
    # [2720, 365, 1634, 1229, 140, 351, 1664, 160, 1534, 1233, 618, 267, 2490, 213, 2483, 89, 273, 665, 352, 222, 2265, 2612, 429, 2492, 2827, 269, 1002, 202, 821, 1615, 1284, 830, 176, 1116, 2626, 1988, 415, 23, 694, 133, 1536, 510, 290, 152, 204, 1034, 1273, 289, 462, 165]
    test_trace = []
    for (oc, ac) in zip(original_trace, another_trace):
        if oc == ac:
            test_trace.append(oc)
        else:
            test_trace.extend([f"del_{oc}", f"add_{ac}"])
    return test_trace


@pytest.fixture(scope="module")
def mock_t_dfa(mock_original_trace):
    return generate_single_accepting_sequence_dfa(mock_original_trace)

@pytest.fixture(scope="module")
def mock_a_dfa(mock_dataset):
    return generate_automata_from_dataset(mock_dataset)

@pytest.fixture(scope="module")
def mock_t_dfa_aug(mock_original_trace):
    # Generate another t_dfa since it's augmented on place
    t_dfa = generate_single_accepting_sequence_dfa(mock_original_trace)
    return augment_trace_automata(t_dfa, items=Items.MOCK)

@pytest.fixture(scope="module")
def mock_a_dfa_aug(mock_dataset, mock_t_dfa):
    a_dfa = generate_automata_from_dataset(mock_dataset, load_if_exists=False)
    return augment_constraint_automata(a_dfa, mock_t_dfa)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(load_path="saved/counterfactual_dataset.pickle")[0]


@pytest.fixture(scope="module")
def automata_gt(dataset):
    good_points, _ = dataset
    return good_points[0][1]


@pytest.fixture(scope="module")
def original_trace(dataset):
    gp, _ = dataset
    return gp[0][0].tolist()

@pytest.fixture(scope="module")
def bad_trace(dataset):
    _, bp = dataset
    return bp[1][0].tolist()

@pytest.fixture(scope="module")
def edited_trace(dataset):
    gp, _ = dataset
    original_trace = gp[0][0].tolist()
    
    # Since we will be doing del_at, add_ot, we can only consider another good
    # trace where ot doesn't appear in the original trace
    trace_chars = set(original_trace)
    for another_trace, _ in gp:
        if len(set(another_trace) & trace_chars) == 0:
            break
    # [2720, 365, 1634, 1229, 140, 3298, 1664, 160, 1534, 1233, 618, 267, 2490, 2492, 2483, 89, 273, 665, 352, 222, 2265, 2612, 429, 213, 2827, 532, 1002, 202, 821, 1615, 1284, 830, 176, 1116, 2626, 23, 415, 1988, 694, 133, 1536, 510, 290, 152, 204, 1770, 1273, 289, 462, 165]
    # [2720, 365, 1634, 1229, 140, 351, 1664, 160, 1534, 1233, 618, 267, 2490, 213, 2483, 89, 273, 665, 352, 222, 2265, 2612, 429, 2492, 2827, 269, 1002, 202, 821, 1615, 1284, 830, 176, 1116, 2626, 1988, 415, 23, 694, 133, 1536, 510, 290, 152, 204, 1034, 1273, 289, 462, 165]
    test_trace = []
    for (oc, ac) in zip(original_trace, another_trace):
        if oc == ac:
            test_trace.append(oc)
        else:
            test_trace.extend([f"del_{oc}", f"add_{ac}"])
    return test_trace

@pytest.fixture(scope="module")
def t_dfa(original_trace):
    return generate_single_accepting_sequence_dfa(original_trace)

@pytest.fixture(scope="module")
def a_dfa(dataset):
    return generate_automata_from_dataset(dataset)

@pytest.fixture(scope="module")
def t_dfa_aug(original_trace):
    t_dfa = generate_single_accepting_sequence_dfa(original_trace)
    return augment_trace_automata(t_dfa)

@pytest.fixture(scope="module")
def a_dfa_aug(dataset, t_dfa):
    a_dfa = generate_automata_from_dataset(dataset, load_if_exists=False)
    return augment_constraint_automata(a_dfa, t_dfa)

@pytest.fixture(scope="module")
def config():
    return get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)

@pytest.fixture(scope="module")
def model(config) -> SequentialRecommender:
    model = generate_model(config)
    return model

@pytest.fixture(scope="module")
def interactions(config):
    return InteractionGenerator(config)

@pytest.fixture(scope="module")
def sequences(config):
    return SequenceGenerator(config)
