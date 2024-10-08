from aalpy.learning_algs import run_RPNI
from aalpy.automata.Dfa import Dfa
from dataset_generator import NumItems
from recommenders.test import model_predict, load_dataset, load_data, generate_model
import pickle
import os
from typing import Union, List
from aalpy.utils.HelperFunctions import make_input_complete
from recbole.config import Config
from tqdm import tqdm
import random

automata_save_path = "automata.pickle"

def generate_automata(dataset) -> Union[None, Dfa]:
    if os.path.exists(automata_save_path):
        print("Loaded existing automata")
        return load_automata()
    print("Existing automata not found, generating a new one based on the provided dataset")
    dfa = run_RPNI(data=dataset, automaton_type="dfa")
    if dfa is None:
        return 
    save_automata(dfa)
    return dfa

def save_automata(automata):
    with open(automata_save_path, "wb") as f:
        pickle.dump(automata, f)

def load_automata():
    with open(automata_save_path, "rb") as f:
        return pickle.load(f)

def test_automata(automata, dataset):
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
    print(f"Automata correctly identifies good and bad points")

def test_automata_against_bb(automata: Dfa, gt: int):
    """
    Test if automata accepts good sequences (model(x) == gt) on the entire test
    set and refuses bad sequences (model(x) != gt) and calculates the
    precision.
    """
    parameter_dict_ml1m = {
            'load_col': {"inter": ['user_id', 'item_id', 'rating', 'timestamp']},
            'train_neg_sample_args': None,
            "eval_batch_size": 1
            }
    config = Config(model='BERT4Rec', dataset='ml-1m', config_dict=parameter_dict_ml1m)
    _, _, test_data = load_data(config)
    model = generate_model(config)
    
    good_predictions, bad_predictions = 0,0
    for _, data in enumerate(tqdm(test_data, "Testing automata against black box model...")):
        interaction = data[0]
        point = interaction.interaction["item_id_list"].squeeze(0)
        # print(point)
        bb_label = model_predict(point, prob=False, default_interaction=interaction, default_model=model)
        automata_accepts = run_automata(automata, point.tolist())
        good_prediction = gt and automata_accepts or bb_label != gt and not automata_accepts
        if good_prediction:
            good_predictions += 1
        else:
            bad_predictions += 1
    print(f"Good predictions: {good_predictions}\nBad predictions: {bad_predictions}")
    print(f"Automata precision: {good_predictions/(good_predictions + bad_predictions)*100}%") #precision is really high
    return good_predictions, bad_predictions



def generate_syntetic_point(min_value:int=1, max_value: NumItems=NumItems.ML_1M.value, length: int = 50):
    point = []
    while len(point) < length:
        item = random.randint(min_value, max_value)
        if item not in point:
            point.append(item)
    return point
    

def run_automata(automata: Dfa, input: List[int]):
    automata.reset_to_initial()
    result = False
    for i, char in enumerate(input):
        try:
            result = automata.step(char)
        except KeyError:
            # print(f"Char {char} at position {i} ignored because not in alphabet")
            continue
    return result

def augment_automata(automata: Dfa):
    pass


if __name__ == "__main__":
    config = {"alphabet_size": NumItems.ML_1M.value, "num_states": 100}
    good_points, bad_points = load_dataset(load_path="saved/counterfactual_dataset.pickle") 
    print(good_points[0])
    alphabet = [i for i in range(config["alphabet_size"])]
    data = [ (seq[0].tolist(), True) for seq in good_points ] + [ (seq[0].tolist(), False) for seq in bad_points ]
    dfa = generate_automata(data)
    if dfa is None:
        raise RuntimeError("DFA is None, aborting")
    dfa = make_input_complete(dfa)
    assert dfa.is_input_complete(), "Dfa is not input complete"
    # print([{state.state_id: state.transitions} for state in dfa.states])
    good_point = [2720,  365, 1634, 1229,  140,  351, 1664,  160, 1534, 1233,  618,  267,
                  2490,  213, 2483,   89,  273,  665,  352,  222, 2265, 2612,  429, 2492,
                  2827,  532, 1002,  202,  821, 1615, 1284,  830,  176, 1116, 2626,   23,
                  415, 1988,  694,  133, 1536,  222,  290,  152,  204, 1034, 1273,  289,
                  462,  165]
    #good label on which the automata was learned on
    automata_gt = good_points[0][1] 
    # run_automata(dfa, good_point)
    # test_automata(dfa, (good_points, bad_points))
    test_automata_against_bb(dfa, gt=automata_gt)

     

