from config.config import ConfigParams
from core.automata_learning.passive_learning import learning_pipeline
from core.automata_learning.utils import run_automata
from core.evaluation.alignment.utils import preprocess_interaction
from core.models.config_utils import generate_model, get_config
from utils.generators import DatasetGenerator
from utils.utils import printd


def test_automata_accepts_source_sequence():
    """
    Test if automata accepts the source sequence
    """
    config = get_config(model=ConfigParams.MODEL, dataset=ConfigParams.DATASET)
    model = generate_model(config)
    i = 0
    ConfigParams.override_params(
        {
            "generation": {"targeted": False, "categorized": False},
            "evolution": {"pop_size": 512, "generations": 5},
        }
    )
    datasets = DatasetGenerator(config=config, return_interaction=True)
    for _ in range(5):
        interaction, dataset = next(datasets)
        trace = preprocess_interaction(interaction)
        assert -1 not in trace
        dfa = learning_pipeline(trace, dataset)
        assert run_automata(dfa, trace), f"Automata do not accept sequence {i}, {trace}"
        printd(f"{i} [PASSED], automata accepts the source trace", level=1)
