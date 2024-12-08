from abc import ABC, abstractmethod

from type_hints import Dataset
from typing import Callable, List
from torch import Tensor


class GenerationStrategy(ABC):
    def __init__(self, 
                 input_seq: Tensor, 
                 model: Callable,
                 alphabet: List[int],
                 good_examples: bool=True,
                 verbose: bool=False):
        self.input_seq = input_seq
        self.model = model
        self.alphabet = alphabet
        self.good_examples = good_examples
        self.verbose = verbose

    @abstractmethod
    def generate(self) -> Dataset:
        pass

    def replace_alphabet(self, alphabet):
        self.alphabet = alphabet
    
