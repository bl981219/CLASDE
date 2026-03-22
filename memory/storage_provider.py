import logging
from typing import Dict, Any, Optional
from memory.experiment_db import ExperimentDatabase
from memory.hypothesis_db import HypothesisDatabase
from memory.literature_db import LiteratureDatabase
from memory.knowledge_graph import KnowledgeGraphMemory

logger = logging.getLogger(__name__)

class StorageRegistry:
    """
    Centralized registry for all database and memory components.
    
    This abstraction allows the CampaignManager to be backend-agnostic,
    facilitating future migrations to SQL or Redis-based storage.
    """
    def __init__(self):
        self.experiment_db = ExperimentDatabase()
        self.hypothesis_db = HypothesisDatabase()
        self.literature_db = LiteratureDatabase()
        self.kg_memory = KnowledgeGraphMemory()
        
        self.is_loaded = False

    def load_all(self):
        """Loads all databases from their respective storage locations."""
        self.experiment_db.load()
        self.hypothesis_db.load()
        self.literature_db.load()
        self.knowledge_graph = self.kg_memory.load()
        self.is_loaded = True
        logger.info("All scientific databases loaded into StorageRegistry.")

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
