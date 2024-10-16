from copy import deepcopy

import pytest
import torch

from automata_learning import learning_pipeline
from automata_utils import invert_automata, run_automata
from graph_search import encode_action_str, print_action
from models.ExtendedBERT4Rec import ExtendedBERT4Rec
from recommenders.generate_dataset import generate_counterfactual_dataset
from trace_alignment import (align, create_intersection_automata,
                             trace_alignment, trace_disalignment)


class TestMockData:
    def test_augmented_trace_automata(self, mock_t_dfa, mock_t_dfa_aug, mock_original_trace, mock_edited_trace):
        # mock_t_dfa.visualize("saved_automatas/mock_t_dfa")
        # mock_t_dfa_aug.visualize("saved_automatas/mock_t_dfa_aug")

        t_dfa_accepts = run_automata(mock_t_dfa, mock_original_trace)
        assert t_dfa_accepts, "T_DFA rejected good point"

        t_dfa_aug_accepts = run_automata(mock_t_dfa_aug, mock_edited_trace)
        assert t_dfa_aug_accepts, "T_DFA rejected edited good point"

    def test_augmented_constraint_automata(self, mock_a_dfa, mock_a_dfa_aug, mock_original_trace, mock_edited_trace): 
        # mock_a_dfa.visualize("saved_automatas/mock_a_dfa")
        # mock_a_dfa_aug.visualize("saved_automatas/mock_a_dfa_aug")

        a_dfa_accepts = run_automata(mock_a_dfa, mock_original_trace)
        assert a_dfa_accepts, "A_DFA rejected good point"

        a_dfa_accepts = run_automata(mock_a_dfa_aug, mock_edited_trace)
        assert a_dfa_accepts, "A_DFA rejected edited good point"

    def test_run_trace_alignment_bad_trace(self, mock_a_dfa_aug, mock_bad_trace):
        a_dfa_aug_accepts = run_automata(mock_a_dfa_aug, mock_bad_trace)
        assert not a_dfa_aug_accepts, f"Bad trace should be accepted"
        
        # TODO: if a character cannot be read by the automata, everything
        # collapses, see how to handle this
        alignment, cost, _= trace_alignment(mock_a_dfa_aug, mock_bad_trace)
        print(f"Best alignment {alignment} with cost {cost}")

    def test_trace_alignment_single(self, mock_a_dfa_aug, mock_bad_trace):
        aligned_trace, _, _ = trace_alignment(mock_a_dfa_aug, mock_bad_trace)
        aligned_accepts = run_automata(mock_a_dfa_aug, aligned_trace)
        print(f"[{self.test_trace_alignment_single.__name__}] Original bad trace: {mock_bad_trace}")
        print(f"[{self.test_trace_alignment_single.__name__}] Aligned bad trace: {aligned_trace}")
        assert aligned_accepts, "Automa should accept aligned trace"
        original_rejects = not run_automata(mock_a_dfa_aug, mock_bad_trace)
        assert original_rejects, "Automa should reject original bad trace"

    def test_trace_alignment(self, mock_a_dfa_aug, mock_dataset):
        _, bp = mock_dataset
        for bad_trace, _ in bp:
            self.test_trace_alignment_single(mock_a_dfa_aug, bad_trace)
        
    def test_trace_disalignment_single(self, mock_a_dfa_aug, mock_original_trace):
        inv_mock_a_dfa_aug = deepcopy(mock_a_dfa_aug)
        invert_automata(inv_mock_a_dfa_aug)
        good_trace_rejects = not run_automata(inv_mock_a_dfa_aug, mock_original_trace)
        assert good_trace_rejects, "Inverted Automa should reject good trace"
        aligned_trace, _, _ = trace_alignment(inv_mock_a_dfa_aug, mock_original_trace)
        print(f"[{self.test_trace_disalignment_single.__name__}] Original trace: {mock_original_trace}")
        print(f"[{self.test_trace_disalignment_single.__name__}] Aligned original trace: {aligned_trace}")
        aligned_accepts = run_automata(inv_mock_a_dfa_aug, aligned_trace)
        assert aligned_accepts, "Inverted Automa should accetps aligned bad trace"

    def test_trace_disalignment(self, mock_a_dfa_aug, mock_dataset):
        gp, _ = mock_dataset
        for good_trace, _ in gp:
            self.test_trace_disalignment_single(mock_a_dfa_aug, good_trace)

@pytest.mark.skip()
class TestRealData:
    def test_augmented_trace_automata(self, t_dfa, t_dfa_aug, original_trace, edited_trace):
        t_dfa_accepts = run_automata(t_dfa, original_trace)
        assert t_dfa_accepts, "T_DFA rejected good point"

        t_dfa_aug_accepts = run_automata(t_dfa_aug, edited_trace)
        assert t_dfa_aug_accepts, "T_DFA rejected edited good point"

    def test_augmented_constraint_automata(self, a_dfa, a_dfa_aug, original_trace, edited_trace): 
        a_dfa_accepts = run_automata(a_dfa, original_trace)
        assert a_dfa_accepts, "A_DFA rejected good point"

        a_dfa_accepts = run_automata(a_dfa_aug, edited_trace)
        assert a_dfa_accepts, "A_DFA rejected edited good point"

    def test_trace_alignment_single(self, a_dfa_aug, bad_trace):
        aligned_trace, _, _= trace_alignment(a_dfa_aug, bad_trace)
        aligned_accepts = run_automata(a_dfa_aug, aligned_trace)
        assert aligned_accepts, "Automa should accept aligned trace"
        original_rejects = not run_automata(a_dfa_aug, bad_trace)
        assert original_rejects, "Automa should reject original bad trace"

    def test_trace_alignment(self, a_dfa_aug, dataset):
        _, bp = dataset
        for bad_trace, _ in bp:
            self.test_trace_alignment_single(a_dfa_aug, bad_trace)

    def test_trace_disalignment_single(self, a_dfa_aug, original_trace):
        inv_mock_a_dfa_aug = deepcopy(a_dfa_aug)
        invert_automata(inv_mock_a_dfa_aug)
        good_trace_rejects = not run_automata(inv_mock_a_dfa_aug, original_trace)
        assert good_trace_rejects, "Inverted Automa should reject good trace"
        aligned_trace, _ = trace_alignment(inv_mock_a_dfa_aug, original_trace)
        print(f"[{self.test_trace_disalignment_single.__name__}] Original trace: {original_trace}")
        print(f"[{self.test_trace_disalignment_single.__name__}] Aligned original trace: {aligned_trace}")
        aligned_accepts = run_automata(inv_mock_a_dfa_aug, aligned_trace)
        assert aligned_accepts, "Inverted Automa should accetps aligned bad trace"

    def test_trace_disalignment(self,a_dfa_aug, dataset):
        gp, _ = dataset
        for good_trace, _ in gp:
            self.test_trace_disalignment_single(a_dfa_aug, good_trace)

class TestUtils:
    def test_align(self):
        # original_trace = [1,2,3,5,6]
        alignment = ("sync_1", "del_2", "add_4", "sync_3", "sync_5", "del_6")
        encoded_alignment = tuple(encode_action_str(a) for a in alignment)
        # print(f"[{test_align.__name__}] encoded_alignment is {encoded_alignment}")
        aligned_trace = align(encoded_alignment)
        # print(f"[{test_align.__name__}] Decoded aligned trace is {aligned_trace}")
        
        correct_alignment = [1,4,3,5]
        assert aligned_trace == correct_alignment, f"""
        Aligned trace is wrong
        corect: {correct_alignment}
        computed: {aligned_trace}
        """

@pytest.mark.incremental
class TestParticularCases:
    """
    This class tests all those particular cases where a bug was found, in order
    to see if the bug is fixed for good. Particular hard coded traces are used.
    """
    def test_mismatch(self, model):
        """
        For some traces, the original sequence is rejected by the dfa (correct),
        and then the found counterfactual is accepted by the dfa (correct), but the
        counterfactual has the same label as the original sequence when inputted
        into the black box model (incorrect)
        """
        trace = torch.tensor([1346, 669, 648, 198, 1315, 1334, 423, 1342, 658,
                              1773, 1380, 1175, 1089, 908, 622, 2892, 3284, 2469,
                              2797, 811, 914, 576, 2885, 2147, 2609, 1834, 2828,
                              1383, 1181, 700, 2206, 182, 1771, 1318, 1867, 1092,
                              2115, 1608, 936, 1057, 1109, 2883, 1826, 2607, 1153,
                              1138, 1204, 2544, 2137, 2055]).unsqueeze(0)
        original_label = model._full_sort_predict_from_sequence(trace).argmax(-1).item()
        train, _ = generate_counterfactual_dataset(trace, model)
        trace = trace.squeeze(0).tolist()
        a_dfa_aug = learning_pipeline(trace, train)
        original_rejects = not run_automata(a_dfa_aug, trace)
        #TODO: the problem is here, meaning that the problem is not the
        #alignment, but the automata learning on this particular trace
        assert original_rejects, "Automata doesn't reject the original sequence"

        a_dfa_aug.reset_to_initial()
        aligned, cost, alignment = trace_disalignment(a_dfa_aug, trace)
        aligned_accepts = run_automata(a_dfa_aug, aligned)
        assert aligned_accepts, "Automata doesn't accept the counterfactual sequence"

        print(f"Alignment is {[print_action(a) for a in alignment]}")
        counter_label = model._full_sort_predict_from_sequence(torch.tensor(aligned).unsqueeze(0)).argmax(-1).item()
        assert original_label != counter_label, f"Original label and counter label are equal: {original_label} == {counter_label}"

    def test_all_sync(self, model):
        """
        For some traces, the counterfactual only does sync. The problem with this
        is the fact that the automata accepts the original sequence, while it
        should be rejecting it. Because of this, the original trace is already a
        counterfactual, which is wrong.
        """
        trace = torch.tensor([3430, 1371, 658, 595, 24, 339, 1238, 2572, 1232, 560,
                              247, 1250, 744, 1451, 2233, 1241, 961, 122, 986,
                              946]).unsqueeze(0)
        train, _ = generate_counterfactual_dataset(trace, model)
        trace = trace.squeeze(0).tolist()
        a_dfa_aug = learning_pipeline(trace, train)
        aligned, cost, alignment = trace_disalignment(a_dfa_aug, trace)
        print(aligned)

