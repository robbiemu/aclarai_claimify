from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseEmbedder(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def embed(self, sentences: List[str]) -> np.ndarray:
        """Converts a list of sentences into a numpy array of embeddings."""
        pass
