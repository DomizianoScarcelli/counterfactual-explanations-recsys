from aalpy.learning_algs import run_RPNI
from recommenders.test import load_dataset


def generate_automata(dataset):
    dfa = run_RPNI(data=dataset, automaton_type="dfa")
    dfa.visualize()
    return dfa

def test_automata(automata, dataset):
    pass


if __name__ == "__main__":
    config = {"alphabet_size": 1700, "num_states": 100}
    good_points, bad_points = load_dataset(load_path="saved/counterfactual_dataset.pickle") 
    alphabet = [i for i in range(config["alphabet_size"])]
    data = [ (seq[0].tolist(), True) for seq in good_points ] + [ (seq[0].tolist(), False) for seq in bad_points ]
    dfa = generate_automata(data)
    pass

