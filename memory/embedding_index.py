import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import os
import json

logger = logging.getLogger(__name__)

class EmbeddingIndex:
    """
    Vector search capability for all scientific memory objects.
    
    Enables semantic similarity searches across:
    - Atomic structures (via feature vectors)
    - Hypotheses (via language embeddings)
    - Literature claims
    """
    def __init__(self, storage_path: str = "data/results/embedding_index.json") -> None:
        self.storage_path = storage_path
        self.embeddings: List[np.ndarray] = [] # List of np.ndarray
        self.metadata: List[Dict[str, Any]] = [] # List of dict pointers to other DBs

    def add_item(self, vector: np.ndarray, item_metadata: Dict[str, Any]) -> None:
        """Index a new vector and its associated metadata."""
        self.embeddings.append(vector)
        self.metadata.append(item_metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Perform a simple cosine similarity search."""
        if not self.embeddings:
            return []
            
        # Guard against zero-norm vectors
        query_norm = np.linalg.norm(query_vector)
        if query_norm < 1e-9:
            logger.warning("Search query vector has near-zero norm. Returning empty.")
            return []

        # Stack embeddings into a matrix
        matrix = np.vstack(self.embeddings)
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        
        # Guard against zero-norm stored embeddings
        matrix_norms[matrix_norms < 1e-9] = 1.0 
        
        # Normalize
        norm_query = query_vector / query_norm
        norm_matrix = matrix / matrix_norms
        
        # Dot product for cosine similarity
        similarities = np.dot(norm_matrix, norm_query)
        
        # Get top indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results: List[Tuple[Dict[str, Any], float]] = []
        for idx in top_indices:
            results.append((self.metadata[idx], float(similarities[idx])))
        return results

    def save(self) -> None:
        """Serialize index."""
        data = {
            "embeddings": [v.tolist() for v in self.embeddings],
            "metadata": self.metadata
        }
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f)

    def load(self) -> None:
        """Load index."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                self.embeddings = [np.array(v) for v in data["embeddings"]]
                self.metadata = data["metadata"]
