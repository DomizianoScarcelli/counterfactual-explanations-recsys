from aalpy.automata.Dfa import Dfa
from dataset_generator import NumItems
from recommenders.test import model_predict, load_dataset, load_data, generate_model
from recbole.config import Config
from tqdm import tqdm
from automata_learning import run_automata, generate_automata_from_dataset, generate_single_accepting_sequence_dfa

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



if __name__ == "__main__":
    #good label on which the automata was learned on
    config = {"alphabet_size": NumItems.ML_1M.value, "num_states": 100}
    good_points, bad_points = load_dataset(load_path="saved/counterfactual_dataset.pickle") 
    automata_gt = (good_points[0][1]).item()
    good_point = good_points[0][0]
    a_dfa = generate_automata_from_dataset((good_points, bad_points))
    t_dfa = generate_single_accepting_sequence_dfa(good_point)
    test_automata(a_dfa, (good_points, bad_points))
    test_automata_against_bb(a_dfa, gt=automata_gt)

