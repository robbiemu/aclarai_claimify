from typing import List
import numpy as np
from .base import BaseEmbedder

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    An embedder that uses the sentence-transformers library.
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError(
                "The 'sentence-transformers' library is required for this embedder. "
                "Please install it with: pip install sentence-transformers"
            )

    def embed(self, sentences: List[str]) -> np.ndarray:
        """
        Embeds a list of sentences using the sentence-transformer model.

        Args:
            sentences: A list of sentences to embed.

        Returns:
            A numpy array of embeddings.
        """
        return self.model.encode(sentences, convert_to_numpy=True)
