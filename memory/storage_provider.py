import logging
from typing import Dict, Any, Optional, List
from memory.experiment_db import ExperimentDatabase
from memory.hypothesis_db import HypothesisDatabase
from memory.literature_db import LiteratureDatabase
from memory.knowledge_graph import KnowledgeGraphMemory

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
        
        self.use_memory = use_memory
        self.is_loaded = False

    def load_all(self):
        """Loads all databases from their respective storage locations."""
        self.experiment_db.load()
        self.hypothesis_db.load()
        self.literature_db.load()
        self.knowledge_graph = self.kg_memory.load()
        self.is_loaded = True
        logger.info(f"All scientific databases loaded. Memory enabled: {self.use_memory}")

    def save_all(self):
        """Persists all databases to disk."""
        self.experiment_db.save()
        self.hypothesis_db.save()
        self.literature_db.save()
        self.kg_memory.save(self.knowledge_graph)
        logger.info("All scientific databases persisted.")

    def get_knowledge_graph(self):
        """Returns the active KnowledgeGraph instance."""
        if not self.is_loaded: self.load_all()
        return self.knowledge_graph

    def retrieve_similar_results(self, query_state: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieval API for agents to find similar historical experiments."""
        if not self.use_memory:
            return []
        
        # Placeholder: In production, use vector search on state embeddings
        # For now, return empty or top results from ExperimentDB
        return self.experiment_db.get_training_data()[:top_k]
