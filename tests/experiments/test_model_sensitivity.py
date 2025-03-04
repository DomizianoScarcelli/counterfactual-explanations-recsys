from statistics import mean

import torch
from tqdm import tqdm

from config.config import ConfigParams
from core.generation.utils import Items, get_items
from core.models.config_utils import generate_model, get_config
from core.models.utils import topk
from type_hints import RecDataset
from utils.utils import jaccard_sim, ndcg_at, precision_at
from utils.utils import SequenceGenerator


def test_ModelSensitivity_YieldsCorrectMetrics_WhenSequencesAreAllEqual():
    position = 49
    config = get_config(dataset=ConfigParams.DATASET, model=ConfigParams.MODEL)
    sequences = SequenceGenerator(config)
    model = generate_model(config)
    
    if ConfigParams.DATASET in [RecDataset.ML_1M, RecDataset.ML_100K]:
        alphabet = torch.tensor(list(get_items()))
    else:
        raise NotImplementedError(f"Dataset {ConfigParams.DATASET} not supported yet!")
    i = 0
    start_i, end_i = 0, 100
    changes, at_least_changes, jaccards = set(), set(), set()
    count = 0
    pbar = tqdm(total=end_i-start_i, desc=f"Testing model sensitivity on position {position}")
    for i, sequence in enumerate(sequences):
        if i < start_i:
            continue
        if i >= end_i:
            break
        pbar.update(1)
        count += 1

        x = sequence
        
        out = model(x)
        x_primes = x.repeat(len(alphabet), 1)
        out_primes = model(x_primes)

        k = 5
        out_k = topk(out, k, dim=-1, indices=True).squeeze() #[K]
        out_primes_k = topk(out_primes, k, dim=-1, indices=True) #[len(alphabet), K]

        assert mean(jaccard_sim(a=out_k, b=(out_prime_k.squeeze() if k != 1 else out_prime_k)) for out_prime_k in out_primes_k) == 1
        assert mean(precision_at(k=k, a=out_k, b=(out_prime_k.squeeze() if k != 1 else out_prime_k)) for out_prime_k in out_primes_k) == 1
        assert mean(ndcg_at(k=k, a=out_k, b=out_prime_k.squeeze() if k!= 1 else out_prime_k) for out_prime_k in out_primes_k) == 1

