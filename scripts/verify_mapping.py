from config.config import ConfigParams
from core.generation.utils import token2id
from type_hints import RecDataset

dataset_target_items = {
    RecDataset.ML_100K: [50, 411, 630, 1305],
    RecDataset.STEAM: [271590, 35140, 292140, 582160],
    RecDataset.ML_1M: [2858, 2005, 728, 2738],
}

dataset_map = {
    RecDataset.ML_1M: "ML_1M",
    RecDataset.STEAM: "STEAM",
    RecDataset.ML_100K: "ML_100K",
}


for dataset, items in dataset_target_items.items():
    ConfigParams.override_params({"settings": {"dataset": dataset_map[dataset]}})
    for item in items:
        print(
            f"Dataset {dataset.value} {item} -> {token2id(dataset=dataset, token=str(item))}"
        )
