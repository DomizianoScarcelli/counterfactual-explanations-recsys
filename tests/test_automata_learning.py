import pytest
from aalpy.automata.Dfa import Dfa
from recbole.config import Config
from tqdm import tqdm

from automata_utils import run_automata
from performance_evaluation.evaluation_utils import compute_metrics, print_confusion_matrix
from recommenders.model_funcs import model_predict
from recommenders.test import generate_model, load_data, load_dataset
from type_hints import Dataset, GoodBadDataset
from typing import Tuple

@pytest.mark.skip()
class TestAutomataPerformances: 
    def test_automata(self, a_dfa: Dfa, dataset: GoodBadDataset):
        """
        Test if automata accepts good sequences on the learning set of good points
        and bad points.
        """
        good_points, bad_points = dataset
        for good_point, bad_point in zip(good_points, bad_points):
            good_result = run_automata(a_dfa, good_point[0].tolist())
            bad_result = run_automata(a_dfa, bad_point[0].tolist())
            assert good_result, f"Wrong result for good point: {good_result}"
            assert not bad_result, f"Wrong result for bad point: {bad_result}"

# @pytest.mark.skip("Don't know if this makes sense")
# def test_automata_against_bb(a_dfa: Dfa, automata_gt: int):
#     """
#     Test the capacity of the automa to approximate the neighbourhood of x
#     described by the black box model. The evaluation is in term of precision,
#     accuracy and recall.
#     """
#     parameter_dict_ml1m = {
#         'load_col': {"inter": ['user_id', 'item_id', 'rating', 'timestamp']},
#         'train_neg_sample_args': None,
#         "eval_batch_size": 1
#     }
#     config = Config(model='BERT4Rec', dataset='ml-1m', config_dict=parameter_dict_ml1m)
#     train_data, valid_data, test_data = load_data(config)
#     model = generate_model(config)
    
#     tp, tn, fp, fn = 0, 0, 0, 0
#     for _, data in enumerate(tqdm(test_data, "Testing automata against black box model...")):
#         interaction = data[0]
#         point = interaction.interaction["item_id_list"].squeeze(0)
#         bb_label = model_predict(seq=point, prob=False, interaction=interaction, model=model)
#         automata_accepts = run_automata(a_dfa, point.tolist())
#         bb_good_point = (bb_label == automata_gt)
#         bb_bad_point = (bb_label != automata_gt)
#         if (bb_good_point and automata_accepts): tp+=1
#         if (bb_bad_point and automata_accepts): fp+=1
#         if (bb_bad_point and not automata_accepts): tn+=1
#         if (bb_good_point and not automata_accepts): fn+=1
    
#     precision, accuracy, recall = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
    
#     print(f"Precision: {precision}")
#     print(f"Recall: {recall}")
#     print(f"Accuracy: {accuracy}")
#     print_confusion_matrix(tp=tp, fp=fp, tn=tn, fn=fn)
#     assert precision > 0.5, f"Automata precision is too low: {precision}"
#     assert recall > 0.5, f"Automata precision is too low: {precision}"
#     assert accuracy > 0.5, f"Automata precision is too low: {precision}"
