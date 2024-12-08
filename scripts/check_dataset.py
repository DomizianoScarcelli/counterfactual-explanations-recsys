import pickle

from recbole.data.dataset.sequential_dataset import SequentialDataset

from genetic.utils import get_category_map

category_map = get_category_map()
category_map_keys = set(int(a) for a in category_map)
print(f"Category mapping length is: {len(category_map_keys)}, with min {min(category_map_keys)} and max {max(category_map_keys)}")


sequential_dataset_path = "data/ml-1m-SequentialDataset.pth"
with open(sequential_dataset_path, "rb") as f:
    dataset:SequentialDataset = pickle.load(f)

# print(dataset.field2token_id)

print("Dataset fields", list(dataset.field2token_id.keys()))
tokens = set(int(a) for a in dataset.field2token_id["item_id"].keys() if a != "[PAD]")
print(f"Tokens len is: {len(tokens)}, with min {min(tokens)} and max {max(tokens)}")
holes = (set(range(min(tokens), max(tokens)+1))) - tokens
print(f"Tokens has holes: {holes}")

# print(dataset.field2id_token["item_id"])
ids = set(range(len(dataset.field2id_token["item_id"])))
print(f"Ids len is: {len(ids)}, with min {min(ids)} and max {max(ids)}")
holes = (set(range(min(ids), max(ids)+1))) - ids
print(f"Ids has holes: {holes}")



