# from genetic.dataset.utils import interaction_to_tensor
# from models.utils import topk
# from utils_classes.generators import InteractionGenerator
# from config import ConfigParams
# from models.config_utils import generate_model, get_config
# from statistics import mean
# from tqdm import tqdm


# conf = get_config(dataset = ConfigParams.DATASET, model=ConfigParams.MODEL)
# model = generate_model(conf)

# interactions = InteractionGenerator(conf, whole_interaction=True, split="test")
# precs = []
# k = 10
# for i, inter_data in enumerate(tqdm(interactions)):
#     gt = inter_data[-1].item()
#     seq = interaction_to_tensor(inter_data[0])

#     pred = model(seq)
    
#     top_k_preds = topk(pred, k=k, dim=-1, indices=True)
#     relevant_in_top_k = (top_k_preds == gt).sum().item()  # Check if gt is in top k
#     prec = relevant_in_top_k / k
#     precs.append(prec)
#     if i % 100 == 0:
#         print(f"Mean Prec@{k}: {mean(precs):.4f}")

# print(f"Mean Prec@{k}: {mean(precs):.4f}")



