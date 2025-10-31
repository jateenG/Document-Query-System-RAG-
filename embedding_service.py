import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        [cite_start]Initialize embedding service with local embedding model. [cite: 9]
        Args:
            [cite_start]model_name: HuggingFace model identifier [cite: 9]
        """
        self.model = SentenceTransformer(model_name)
        [cite_start]self.embedding_dim = self.model.get_sentence_embedding_dimension() [cite: 10]
        logger.info(f"Loaded embedding model: {model_name} ({self.embedding_dim} dims)")

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True, batch_size=32)