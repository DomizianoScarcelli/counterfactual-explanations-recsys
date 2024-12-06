from abc import ABC, abstractmethod

from type_hints import Dataset


class GenerationStrategy(ABC):
    @abstractmethod
    def generate(self) -> Dataset:
        pass
    
    #TODO: put clean and postprocess all in generate, remove this from the abstract class
    @abstractmethod
    def clean(self, examples: Dataset) -> Dataset:
        pass

    @abstractmethod
    def postprocess(self, population: Dataset) -> Dataset:
        pass
