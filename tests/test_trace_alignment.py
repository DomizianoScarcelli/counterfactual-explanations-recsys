import warnings
from copy import deepcopy

import pytest
import torch
from aalpy.automata.Dfa import Dfa

from alignment.actions import encode_action_str, print_action
from alignment.alignment import align, trace_alignment, trace_disalignment
from automata_learning.learning import learning_pipeline
from automata_learning.utils import invert_automata, run_automata
from constants import MAX_LENGTH
from generation.dataset.generate import generate
from generation.dataset.utils import get_dataset_alphabet
from models.utils import pad, trim
from type_hints import GoodBadDataset, Trace
from utils_classes.Split import Split


@pytest.mark.skip()
class TestMockData:
    def test_augmented_trace_automata(
        self, mock_t_dfa, mock_t_dfa_aug, mock_original_trace, mock_edited_trace
    ):
        # mock_t_dfa.visualize("saved_automatas/mock_t_dfa")
        # mock_t_dfa_aug.visualize("saved_automatas/mock_t_dfa_aug")
        t_dfa_accepts = run_automata(mock_t_dfa, mock_original_trace)
        assert t_dfa_accepts, "T_DFA rejected good point"

        t_dfa_aug_accepts = run_automata(mock_t_dfa_aug, mock_edited_trace)
        assert t_dfa_aug_accepts, "T_DFA rejected edited good point"

    def test_augmented_constraint_automata(
        self, mock_a_dfa, mock_a_dfa_aug, mock_original_trace, mock_edited_trace
    ):
        # mock_a_dfa.visualize("saved_automatas/mock_a_dfa")
        # mock_a_dfa_aug.visualize("saved_automatas/mock_a_dfa_aug")

        a_dfa_accepts = run_automata(mock_a_dfa, mock_original_trace)
        assert a_dfa_accepts, "A_DFA rejected good point"

        a_dfa_accepts = run_automata(mock_a_dfa_aug, mock_edited_trace)
        assert a_dfa_accepts, "A_DFA rejected edited good point"

    def test_run_trace_alignment_bad_trace(self, mock_a_dfa_aug, mock_bad_trace):
        a_dfa_aug_accepts = run_automata(mock_a_dfa_aug, mock_bad_trace)
        assert not a_dfa_aug_accepts, "Bad trace should be accepted"

        # TODO: if a character cannot be read by the automata, everything
        # collapses, see how to handle this
        alignment, cost, _ = trace_alignment(mock_a_dfa_aug, ([], mock_bad_trace, []))
        print(f"Best alignment {alignment} with cost {cost}")

    @pytest.mark.heavy
    def test_trace_alignment_single(self, mock_a_dfa_aug, mock_bad_trace):
        aligned_trace, _, _ = trace_alignment(mock_a_dfa_aug, ([], mock_bad_trace, []))
        aligned_accepts = run_automata(mock_a_dfa_aug, aligned_trace)
        print(
            f"[{self.test_trace_alignment_single.__name__}] Original bad trace: {
              mock_bad_trace}"
        )
        print(
            f"[{self.test_trace_alignment_single.__name__}] Aligned bad trace: {
              aligned_trace}"
        )
        assert aligned_accepts, "Automa should accept aligned trace"
        original_rejects = not run_automata(mock_a_dfa_aug, mock_bad_trace)
        assert original_rejects, "Automa should reject original bad trace"

    @pytest.mark.heavy
    def test_trace_alignment(self, mock_a_dfa_aug, mock_dataset):
        _, bp = mock_dataset
        for bad_trace, _ in bp:
            self.test_trace_alignment_single(mock_a_dfa_aug, bad_trace)

    @pytest.mark.heavy
    def test_trace_disalignment_single(self, mock_a_dfa_aug, mock_original_trace):
        inv_mock_a_dfa_aug = deepcopy(mock_a_dfa_aug)
        invert_automata(inv_mock_a_dfa_aug)
        good_trace_rejects = not run_automata(inv_mock_a_dfa_aug, mock_original_trace)
        assert good_trace_rejects, "Inverted Automa should reject good trace"
        aligned_trace, _, _ = trace_alignment(inv_mock_a_dfa_aug, mock_original_trace)
        print(
            f"[{self.test_trace_disalignment_single.__name__}] Original trace: {
              mock_original_trace}"
        )
        print(
            f"[{self.test_trace_disalignment_single.__name__}] Aligned original trace: {
              aligned_trace}"
        )
        aligned_accepts = run_automata(inv_mock_a_dfa_aug, aligned_trace)
        assert aligned_accepts, "Inverted Automa should accetps aligned bad trace"

    @pytest.mark.heavy
    def test_trace_disalignment(self, mock_a_dfa_aug, mock_dataset):
        gp, _ = mock_dataset
        for good_trace, _ in gp:
            self.test_trace_disalignment_single(mock_a_dfa_aug, good_trace)


@pytest.mark.skip()
class TestMockSubsequence:
    # def test_trace_disalignment_single(self, mock_a_dfa_aug):
    #     expected = [1, 5, 4, 2, 3]
    #     trace_split = ([1], [5], [2, 3])
    #     aligned_trace, _, _ = trace_disalignment(mock_a_dfa_aug, trace_split)
    #     print(aligned_trace)
    #     assert aligned_trace == expected, f"Aligned trace is not equal to expected: {
    #         aligned_trace} != {expected}"

    def test_subtrace_disalignment(self, mock_a_dfa_aug):
        """
        If an alignment, exists for the entire trace, then it should be found
        by the algorithm. If an alignment exist for a subtrace, then it should
        be found by the algorithm.
        """
        expected = [[5, 4, 1], [5, 2, 4, 1], [5, 4, 1, 2]]
        trace_1 = ([], [5, 3], [4, 1])
        trace_2 = [5, 3, 4, 1]
        # I'm doing alignment (over disalignment) because I designed the trace to work on the original automa
        aligned_trace2, _, _ = trace_alignment(mock_a_dfa_aug, trace_2)
        aligned_trace, _, _ = trace_alignment(mock_a_dfa_aug, trace_1)
        print(f"Aligned Trace 1: {aligned_trace}")
        print(f"Aligned Trace 2: {aligned_trace2}")
        assert (
            aligned_trace in expected
        ), f"Aligned trace is not equal to expected: {
            aligned_trace} not in {expected}"
        assert (
            aligned_trace2 in expected
        ), f"Aligned trace 2 is not equal to expected: {
            aligned_trace2} not in {expected}"


class TestWellFormedRealData:
    def test_well_formed_constraint_automata(self, a_dfa, dataset):
        dataset_alphabet = get_dataset_alphabet(dataset)
        automa_alphabet = set(a_dfa.get_input_alphabet())
        assert dataset_alphabet == automa_alphabet

    def test_well_formed_aug_constraint_automata(
        self, a_dfa_aug: Dfa, original_trace: Trace, dataset: GoodBadDataset
    ):
        """
        Tests if the augmented constraint automata a_dfa_aug is well formed.
        For A+ to be well formed it has to have a del_s for each s in the
        original trace, and an add_s for each s in the a_dfa (A dfa).

        Args:
            a_dfa_aug: The augmented A automata A+
            original_trace: The source trace which generates the dataset, A and A+ dfa
            dataset: The dataset of good and bad examples generated by the original trace
        """
        dataset_alphabet = get_dataset_alphabet(dataset)
        automa_alphabet = set(a_dfa_aug.get_input_alphabet())
        original_trace_symbols = set(original_trace)
        del_syms = set(
            int(s.split("_")[-1]) for s in automa_alphabet if "del" in str(s)
        )
        add_syms = set(
            int(s.split("_")[-1]) for s in automa_alphabet if "add" in str(s)
        )
        sync_syms = set(s for s in automa_alphabet if isinstance(s, int))
        print(len(del_syms), len(add_syms), len(sync_syms))
        assert original_trace_symbols == del_syms
        assert add_syms == sync_syms
        assert sync_syms == dataset_alphabet


@pytest.mark.skip()
class TestRealData:
    @pytest.mark.skip()
    def test_augmented_trace_automata(
        self, t_dfa, t_dfa_aug, original_trace, edited_trace
    ):
        t_dfa_accepts = run_automata(t_dfa, original_trace)
        assert t_dfa_accepts, "T_DFA rejected good point"

        t_dfa_aug_accepts = run_automata(t_dfa_aug, edited_trace)
        assert t_dfa_aug_accepts, "T_DFA rejected edited good point"

    def test_augmented_constraint_automata(
        self, a_dfa, a_dfa_aug, original_trace, edited_trace
    ):
        a_dfa_accepts = run_automata(a_dfa, original_trace)
        assert a_dfa_accepts, "A_DFA rejected good point"

        a_dfa_accepts = run_automata(a_dfa_aug, edited_trace)
        assert a_dfa_accepts, "A_DFA rejected edited good point"

    @pytest.mark.skip()
    def test_trace_alignment_single(self, a_dfa_aug, bad_trace):
        aligned_trace, _, _ = trace_alignment(a_dfa_aug, bad_trace)
        aligned_accepts = run_automata(a_dfa_aug, aligned_trace)
        assert aligned_accepts, "Automa should accept aligned trace"
        original_rejects = not run_automata(a_dfa_aug, bad_trace)
        assert original_rejects, "Automa should reject original bad trace"

    @pytest.mark.skip()
    def test_trace_alignment(self, a_dfa_aug, dataset):
        _, bp = dataset
        for bad_trace, _ in bp:
            self.test_trace_alignment_single(a_dfa_aug, bad_trace)

    @pytest.mark.skip()
    def test_trace_disalignment_single(self, a_dfa_aug, original_trace):
        inv_mock_a_dfa_aug = deepcopy(a_dfa_aug)
        invert_automata(inv_mock_a_dfa_aug)
        good_trace_rejects = not run_automata(inv_mock_a_dfa_aug, original_trace)
        assert good_trace_rejects, "Inverted Automa should reject good trace"
        aligned_trace, _ = trace_alignment(inv_mock_a_dfa_aug, original_trace)
        print(
            f"[{self.test_trace_disalignment_single.__name__}] Original trace: {
              original_trace}"
        )
        print(
            f"[{self.test_trace_disalignment_single.__name__}] Aligned original trace: {
              aligned_trace}"
        )
        aligned_accepts = run_automata(inv_mock_a_dfa_aug, aligned_trace)
        assert aligned_accepts, "Inverted Automa should accetps aligned bad trace"

    @pytest.mark.skip()
    def test_trace_disalignment(self, a_dfa_aug, dataset):
        gp, _ = dataset
        for good_trace, _ in gp:
            self.test_trace_disalignment_single(a_dfa_aug, good_trace)


@pytest.mark.skip()
class TestUtils:
    def test_align(self):
        # original_trace = [1,2,3,5,6]
        alignment = ("sync_1", "del_2", "add_4", "sync_3", "sync_5", "del_6")
        encoded_alignment = tuple(encode_action_str(a) for a in alignment)
        # print(f"[{test_align.__name__}] encoded_alignment is {encoded_alignment}")
        aligned_trace = align(encoded_alignment)
        # print(f"[{test_align.__name__}] Decoded aligned trace is {aligned_trace}")

        correct_alignment = [1, 4, 3, 5]
        assert (
            aligned_trace == correct_alignment
        ), f"""
        Aligned trace is wrong
        corect: {correct_alignment}
        computed: {aligned_trace}
        """


class TestEdgeCases:
    """
    This class tests all those edge cases where a bug was found, in order
    to see if the bug is fixed for good. Particular hard coded traces are used.
    """

    def test_seq_subseq_equality(self, model):
        trace = torch.tensor(
            [
                1346,
                669,
                648,
                198,
                1315,
                1334,
                423,
                1342,
                658,
                1773,
                1380,
                1175,
                1089,
                908,
                622,
                2892,
                3284,
                2469,
                2797,
                811,
                914,
                576,
                2885,
                2147,
                2609,
                1834,
                2828,
                1383,
                1181,
                700,
                2206,
                182,
                1771,
                1318,
                1867,
                1092,
                2115,
                1608,
                936,
                1057,
                1109,
                2883,
                1826,
                2607,
                1153,
                1138,
                1204,
                2544,
                2137,
                2055,
            ]
        ).unsqueeze(0)

        dataset = generate(trace, model)
        original_label = model(trace).argmax(-1).item()
        print(f"Original trace's label: {original_label}")
        trace = trace.squeeze(0).tolist()
        dfa = learning_pipeline(trace, dataset)
        # original_accepts = run_automata(dfa, trace)
        # if not original_accepts:
        #     warnings.warn("Original Automata doesn't accept the original sequence")
        # counterfactual, _, alignment = trace_disalignment(dfa, trace)
        # print(f"Alignment is: {[print_action(a) for a in alignment]}")
        # counterfactual_rejects = not run_automata(dfa, counterfactual)
        # if not counterfactual_rejects:
        #     warnings.warn("Original Automata doesn't reject the counterfactual sequence")

        split = Split(20 / 50, 28 / 50, 2 / 50)
        splitted = split.apply(trace)
        splitted_counterfactual, _, splitted_alignment = trace_disalignment(
            dfa, splitted
        )
        print(
            f"Splitted counterfactual before padding: {torch.tensor(torch.tensor(splitted_counterfactual))}"
        )

        if len(splitted_counterfactual) < MAX_LENGTH:
            splitted_counterfactual = pad(
                trim(torch.tensor(splitted_counterfactual)), MAX_LENGTH
            )

        print(
            f"Splitted counterfactual: {torch.tensor(splitted_counterfactual).unsqueeze(0)}"
        )

        counterfactual_label = (
            model(
                torch.tensor(splitted_counterfactual).unsqueeze(0)
            )
            .argmax(-1)
            .item()
        )

        print(f"Counterfactual trace's label: {counterfactual_label}")

        counterfactual_rejects = not run_automata(dfa, splitted_counterfactual)
        if not counterfactual_rejects:
            warnings.warn(
                "Original Automata doesn't reject the splitted counterfactual sequence"
            )
        assert (
            original_label != counterfactual_label
        ), f"Labels are equal: {original_label} == {counterfactual_label}"

    def test_all_syncs(self, model):
        trace = torch.tensor(
            [
                578,
                65,
                28,
                1432,
                2079,
                199,
                1043,
                1713,
                80,
                63,
                265,
                44,
                152,
                157,
                1059,
                133,
                93,
                49,
                631,
                433,
                190,
                134,
                844,
                79,
                118,
                105,
                639,
                1396,
                51,
                117,
                90,
                21,
                402,
                89,
                336,
            ]
        ).unsqueeze(0)
        dataset = generate(trace, model)
        g, b = dataset
        print(f"Good points: {len(g)}")
        print(f"Bad points: {len(b)}")
        original_label = model(trace).argmax(-1).item()
        print(f"Original trace's label: {original_label}")
        trace = trace.squeeze(0).tolist()
        a_dfa_aug = learning_pipeline(trace, dataset)
        original_accepts = run_automata(a_dfa_aug, trace)
        assert (
            original_accepts
        ), "Original Automata doesn't accept the original sequence"
        disaligned, _, alignment = trace_disalignment(a_dfa_aug, trace)
        print(f"Alignment is: {[print_action(a) for a in alignment]}")
        disaligned_rejects = run_automata(a_dfa_aug, disaligned)
        assert (
            disaligned_rejects
        ), "Original Automata doesn't reject the counterfactual sequence"

    def test_dfa_not_rejecting(self, model):
        """
        For some traces, they are not rejected by the DFA (incorrect), and
        because of this the counterfactual is equal to the original sequence.

        The reason is due by the fact that the generation algorithm fails to
        generate a high percentace of good examples.
        """
        trace = torch.tensor(
            [
                578,
                65,
                28,
                1432,
                2079,
                199,
                1043,
                1713,
                80,
                63,
                265,
                44,
                152,
                157,
                1059,
                133,
                93,
                49,
                631,
                433,
                190,
                134,
                844,
                79,
                118,
                105,
                639,
                1396,
                51,
                117,
                90,
                21,
                402,
                89,
                336,
            ]
        ).unsqueeze(0)
        original_label = model(trace).argmax(-1).item()
        dataset = generate(trace, model)

        trace = trace.squeeze(0).tolist()
        a_dfa_aug = learning_pipeline(trace, dataset)
        original_accepts = run_automata(a_dfa_aug, trace)
        assert (
            original_accepts
        ), "Original Automata doesn't accept the original sequence"

    def test_mismatch(self, model):
        """
        For some traces, the original sequence is rejected by the dfa (correct),
        and then the found counterfactual is accepted by the dfa (correct), but the
        counterfactual has the same label as the original sequence when inputted
        into the black box model (incorrect)
        """
        trace = torch.tensor(
            [
                1346,
                669,
                648,
                198,
                1315,
                1334,
                423,
                1342,
                658,
                1773,
                1380,
                1175,
                1089,
                908,
                622,
                2892,
                3284,
                2469,
                2797,
                811,
                914,
                576,
                2885,
                2147,
                2609,
                1834,
                2828,
                1383,
                1181,
                700,
                2206,
                182,
                1771,
                1318,
                1867,
                1092,
                2115,
                1608,
                936,
                1057,
                1109,
                2883,
                1826,
                2607,
                1153,
                1138,
                1204,
                2544,
                2137,
                2055,
            ]
        ).unsqueeze(0)
        original_label = model(trace).argmax(-1).item()
        dataset = generate(trace, model)

        trace = trace.squeeze(0).tolist()
        a_dfa_aug = learning_pipeline(trace, dataset)
        original_accepts = run_automata(a_dfa_aug, trace)
        assert (
            original_accepts
        ), "Original Automata doesn't accept the original sequence"

        a_dfa_aug.reset_to_initial()
        aligned, _, alignment = trace_disalignment(a_dfa_aug, trace)
        a_dfa_aug.reset_to_initial()
        counterfactual_rejects = not run_automata(a_dfa_aug, aligned)
        assert (
            counterfactual_rejects
        ), "Automata doesn't reject the counterfactual sequence"

        print(f"Alignment is {[print_action(a) for a in alignment]}")
        counter_label = (
            model(torch.tensor(aligned).unsqueeze(0))
            .argmax(-1)
            .item()
        )
        assert (
            original_label != counter_label
        ), f"Original label and counter label are equal: {
            original_label} == {counter_label}"
