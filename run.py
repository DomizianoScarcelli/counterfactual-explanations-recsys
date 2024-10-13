from trace_alignment import trace_alignment, trace_disalignment
from automata_learning import learning_pipeline
from recommenders.generate_dataset import (dataset_generator, get_config,
                                           interaction_generator,
                                           get_sequence_from_interaction)
from type_hints import (Dataset, RecModel, RecDataset, LabeledTensor)
import fire
from torch import Tensor
import time


def main(dataset:RecDataset=RecDataset.ML_1M, 
         model:RecModel=RecModel.BERT4Rec, 
         num_counterfactuals: int=1,
         num_generations: int=20,
         dataset_examples: int=2000):
    print(f"""
          -----------------------
          PARAMETERS
          -----------------------

          ---Model--------------
          dataset: {dataset}
          model: {model}
          num_counterfactuals: {num_counterfactuals}

          ---Genetic Algorithm---
          num_generations: {num_generations}
          dataset_examples: {num_generations}
          -----------------------
          """)
    config = get_config(dataset=dataset, model=model)
    interactions = interaction_generator(config)
    datasets = dataset_generator(config=config)
    for i, (_dataset, interaction) in enumerate(zip(datasets, interactions)):
        start = time.time()
        if i >= num_counterfactuals:
            print(f"Generated {num_counterfactuals}, exiting...")
            break
        source_sequence = get_sequence_from_interaction(interaction)
        if isinstance(source_sequence, Tensor):
            print("Converting source_sequence from Tensor to list")
            source_sequence = source_sequence.tolist()[0]
        dfa = learning_pipeline(source=source_sequence, dataset=_dataset)
        aligned, cost = trace_disalignment(dfa, source_sequence)
        end = time.time()
        generation_time = end-start
        print(f"ALIGNED! {source_sequence} -> {aligned}, C={cost}, Generation time: {generation_time} sec")
        
if __name__ == '__main__':
  fire.Fire(main)
