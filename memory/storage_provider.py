import logging
import numpy as np
from typing import Dict, Any, Optional, List
from memory.experiment_db import ExperimentDatabase
from memory.hypothesis_db import HypothesisDatabase
from memory.literature_db import LiteratureDatabase
from memory.knowledge_graph import KnowledgeGraphMemory
from memory.embedding_index import EmbeddingIndex
from core.state import SurfaceState

logger = logging.getLogger(__name__)

class StorageRegistry:
    """
    Centralized registry for all database and memory components.
    """
    def __init__(self, use_memory: bool = True):
        self.experiment_db = ExperimentDatabase()
        self.hypothesis_db = HypothesisDatabase()
        self.literature_db = LiteratureDatabase()
        self.kg_memory = KnowledgeGraphMemory()
        self.vector_index = EmbeddingIndex()
        
        self.use_memory = use_memory
        self.is_loaded = False

    def load_all(self):
        """Loads all databases from their respective storage locations."""
        self.experiment_db.load()
        self.hypothesis_db.load()
        self.literature_db.load()
        self.vector_index.load()
        self.knowledge_graph = self.kg_memory.load()
        
        # Build index if empty and data exists
        data = self.experiment_db.get_training_data()
        if not self.vector_index.embeddings and data:
            logger.info("Rebuilding vector index from experiment database...")
            for i, entry in enumerate(data):
                state = entry['state']
                if isinstance(state, SurfaceState):
                    feat = np.array(state.get_feature_vector())
                    self.vector_index.add_item(feat, {"db_index": i})
        
        self.is_loaded = True
        logger.info(f"All scientific databases loaded. Memory enabled: {self.use_memory}")

    def save_all(self):
        """Persists all databases to disk."""
        self.experiment_db.save()
        self.hypothesis_db.save()
        self.literature_db.save()
        self.vector_index.save()
        self.kg_memory.save(self.knowledge_graph)
        logger.info("All scientific databases persisted.")

    def get_knowledge_graph(self):
        """Returns the active KnowledgeGraph instance."""
        if not self.is_loaded: self.load_all()
        return self.knowledge_graph

    def retrieve_similar_results(self, query: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieval API for agents to find similar historical experiments.
        Uses vector similarity if query is a SurfaceState.
        """
        if not self.use_memory or not self.is_loaded:
            return []
        
        data = self.experiment_db.get_training_data()
        if not data:
            return []

        # Case 1: Semantic search via feature vectors
        if isinstance(query, SurfaceState):
            feat = np.array(query.get_feature_vector())
            hits = self.vector_index.search(feat, top_k=top_k)
            return [data[hit[0]["db_index"]] for hit in hits]
            
        # Case 2: String query (Keyword-ish fallback or just recent)
        if isinstance(query, str):
            query_lower = query.lower()
            results = []
            for entry in data:
                # Check for simple keyword matches in summary or intuition
                if query_lower in entry['state'].get_summary().lower():
                    results.append(entry)
            if results:
                return results[:top_k]

        # Default fallback: Top-K most recent (last ones in DB)
        return data[-top_k:][::-1]
