# import warnings

# from aalpy.automata.Dfa import Dfa
# from torch._prims_common import number_type

# from automata_learning.learning import learning_pipeline
# from automata_learning.utils import run_automata
# from genetic.dataset.generate import dataset_generator, interaction_generator
# from genetic.dataset.utils import get_sequence_from_interaction
# from models.config_utils import get_config
# from performance_evaluation.evaluation_utils import (compute_metrics,
#                                                      print_confusion_matrix)
# from type_hints import Dataset, GoodBadDataset, RecDataset, RecModel
# from utils_classes.generators import DatasetGenerator

# warnings.simplefilter(action='ignore', category=FutureWarning)

# def single_evaluation(dfa: Dfa, test_dataset: GoodBadDataset):
#     """Evaluate the DFA on test dataset and return evaluation metrics.

#     Args:
#         dfa: The DFA learned on the train dataset
#         test_dataset: The test dataset related to the train dataset
#     """
#     tp, fp, tn, fn = 0,0,0,0
#     good_skipped, bad_skipped = 0,0
#     for (good, bad) in zip(*test_dataset):
#         good_point, bad_point = good[0].squeeze(0).tolist(), bad[0].squeeze(0).tolist()
#         try:
#             dfa_accepts_good = run_automata(dfa, good_point)
#         except KeyError as e:
#             print(f"Good Point not recognized by automata: {e}")
#             good_skipped += 1
#             continue

#         try:
#             dfa_rejects_bad = not run_automata(dfa, bad_point)
#         except KeyError as e:
#             print(f"Bad Point not recognized by automata: {e}")
#             bad_skipped += 1
#             continue

#         if dfa_accepts_good:
#             tp += 1
#         if dfa_rejects_bad:
#             tn += 1
#         if not dfa_accepts_good:
#             fn += 1
#         if not dfa_rejects_bad:
#             fp += 1

#     precision, accuracy, recall = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
#     print(f"Precision: {precision}")
#     print(f"Accuracy: {accuracy}")
#     print(f"Recall: {recall}")
#     total = tp + tn + fn + fp
#     print(f"Skipped: {good_skipped + bad_skipped} over total: {total}")
#     print_confusion_matrix(tp=tp, fp=fp, tn=tn, fn=fn)


# def evaluate_automata_learning():
#     config = get_config(dataset=RecDataset.ML_1M, model=RecModel.BERT4Rec)
#     datasets = DatasetGenerator(config=config, use_cache=True, return_interaction=True)
#     num_tests = 10
#     for idx, (interaction, (train, test)) in enumerate(zip(interactions, datasets)):
#         if idx == num_tests:
#             print(f"Executed {num_tests} tests, exiting...")
#         source_sequence = get_sequence_from_interaction(interaction).squeeze(0).tolist()
#         dfa = learning_pipeline(source_sequence, train)
#         single_evaluation(dfa, test)

# if __name__ == "__main__": 
#     evaluate_automata_learning()
