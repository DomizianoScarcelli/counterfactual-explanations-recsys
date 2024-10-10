import pytest
from aalpy.automata.Dfa import Dfa
from dataset_generator import NumItems
from recommenders.test import model_predict, load_dataset, load_data, generate_model
from recbole.config import Config
from tqdm import tqdm
from automata_learning import (
    run_automata, 
    generate_automata_from_dataset, 
    generate_single_accepting_sequence_dfa, 
)

# Fixtures
@pytest.fixture
def dataset():
    return load_dataset(load_path="saved/counterfactual_dataset.pickle")

@pytest.fixture
def automata(dataset):
    good_points, bad_points = dataset
    return generate_automata_from_dataset((good_points, bad_points))

@pytest.fixture
def automata_gt(dataset):
    good_points, _ = dataset
    return good_points[0][1]

@pytest.fixture
def good_point(dataset):
    good_points, _ = dataset
    return good_points[0][0]

# Test functions
def test_automata(automata: Dfa, dataset):
    """
    Test if automata accepts good sequences on the learning set of good points
    and bad points.
    """
    good_points, bad_points = dataset
    for good_point, bad_point in zip(good_points, bad_points):
        good_result = run_automata(automata, good_point[0].tolist())
        bad_result = run_automata(automata, bad_point[0].tolist())
        assert good_result, f"Wrong result for good point: {good_result}"
        assert not bad_result, f"Wrong result for bad point: {bad_result}"

def test_automata_against_bb(automata: Dfa, automata_gt: int):
    """
    Test the capacity of the automa to approximate the neighbourhood of x
    described by the black box model. The evaluation is in term of precision,
    accuracy and recall.
    """
    parameter_dict_ml1m = {
        'load_col': {"inter": ['user_id', 'item_id', 'rating', 'timestamp']},
        'train_neg_sample_args': None,
        "eval_batch_size": 1
    }
    config = Config(model='BERT4Rec', dataset='ml-1m', config_dict=parameter_dict_ml1m)
    train_data, valid_data, test_data = load_data(config)
    model = generate_model(config)
    
    tp, tn, fp, fn = 0, 0, 0, 0
    for _, data in enumerate(tqdm(test_data, "Testing automata against black box model...")):
        interaction = data[0]
        point = interaction.interaction["item_id_list"].squeeze(0)
        bb_label = model_predict(point, prob=False, default_interaction=interaction, default_model=model)
        automata_accepts = run_automata(automata, point.tolist())
        bb_good_point = (bb_label == automata_gt)
        bb_bad_point = (bb_label != automata_gt)
        if (bb_good_point and automata_accepts): tp+=1
        if (bb_bad_point and automata_accepts): fp+=1
        if (bb_bad_point and not automata_accepts): tn+=1
        if (bb_good_point and not automata_accepts): fn+=1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")
    print(f"""
    Confusion matrix:
    ---------------
    | TP: {tp}  | FP: {fp} |
    ---------------
    | FN: {fn}  | TN: {tn} |
    ---------------
    """)
    assert precision > 0.5, f"Automata precision is too low: {precision}"
    assert recall > 0.5, f"Automata precision is too low: {precision}"
    assert accuracy > 0.5, f"Automata precision is too low: {precision}"
