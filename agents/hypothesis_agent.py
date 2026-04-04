import logging
import numpy as np
import uuid
import json
from typing import List, Dict, Any, Optional
from pydantic import ValidationError
from core.hypothesis import Hypothesis, HypothesisType, HypothesisResult
from science.experiment_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

class HypothesisAgent:
    """
    Agent 1 — The Principal Investigator (PI).
    
    The creative and critical lead. Formulates hypotheses from literature 
    and determines if they are supported by empirical data.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph, hypothesis_db: Any, llm_client: Optional[Any] = None):
        self.kg = knowledge_graph
        self.hyp_db = hypothesis_db
        self.llm = llm_client

    def formulate_initial_hypothesis(self, literature_claims: List[str], research_goal: str) -> Hypothesis:
        """
        Synthesizes prior knowledge and user intent into a testable hypothesis.
        """
        logger.info("[PI] Reviewing literature and formulating initial hypothesis...")
        
        # In a real system, we'd call the LLM here.
        # For now, we simulate the LLM output.
        simulated_response = {
            "id": str(uuid.uuid4()),
            "description": f"Surface cation arrangement determines {research_goal}.",
            "type": "mechanism",
            "predicted_effect": "optimize adsorption energy",
            "target_metric": "adsorption_energy",
            "direction": "nonlinear",
            "falsification_condition": "no correlation found between cation site and energy",
            "source": "LLM",
            "confidence": 0.6
        }

        try:
            hypothesis = Hypothesis(**simulated_response)
        except ValidationError as e:
            logger.error(f"Failed to validate hypothesis: {e}")
            # In production, trigger retry logic here
            raise

        self.hyp_db.add_hypothesis(
            hypothesis=hypothesis.description,
            evidence_ids=[], 
            confidence=hypothesis.confidence
        )
        return hypothesis

    def evaluate_hypothesis(self, hypothesis: Hypothesis, dataset: List[Dict[str, Any]]) -> HypothesisResult:
        """
        Compares experimental results against the hypothesis to produce a formal result.
        """
        if len(dataset) < 2:
            return HypothesisResult(
                hypothesis_id=hypothesis.id,
                supported=False,
                effect_size=0.0,
                confidence=hypothesis.confidence,
                summary="Insufficient data for evaluation."
            )

        logger.info(f"[PI] Evaluating Hypothesis: {hypothesis.description}")
        
        rewards = [d.get("reward", -1e9) for d in dataset if d.get("reward") is not None]
        if not rewards:
             return HypothesisResult(
                hypothesis_id=hypothesis.id,
                supported=False,
                effect_size=0.0,
                confidence=0.0,
                summary="No valid rewards found."
            )
            
        initial_reward = rewards[0]
        max_reward = max(rewards)
        improvement = max_reward - initial_reward
        
        supported = improvement > 0.1
        
        return HypothesisResult(
            hypothesis_id=hypothesis.id,
            supported=supported,
            effect_size=float(improvement),
            confidence=0.8 if supported else 0.3,
            summary=f"Improvement of {improvement:.4f} eV found." if supported else "No significant improvement found."
        )

    def evolve_hypothesis(self, old_hyp: Hypothesis, dataset: List[Dict[str, Any]]) -> Hypothesis:
        """Generates the next hypothesis based on the feedback loop discovery."""
        rewards = [d.get("reward", -1e9) for d in dataset]
        best_idx = int(np.argmax(rewards))
        best_state = dataset[best_idx]['state']
        
        comp_summary = "".join([f"{k}{v}" for k, v in best_state.bulk_composition.items()])
        
        new_description = f"The {comp_summary} arrangement is stable; Oxygen vacancy concentration further stabilizes it."
        
        new_hyp = Hypothesis(
            id=str(uuid.uuid4()),
            description=new_description,
            type=HypothesisType.PRIOR,
            predicted_effect="increase stability",
            target_metric="stability",
            direction="increase",
            falsification_condition="vacancy count does not correlate with stability",
            source="LLM",
            confidence=0.7
        )
        
        self.hyp_db.add_hypothesis(
            hypothesis=new_hyp.description,
            evidence_ids=[best_state.get_id()], 
            confidence=new_hyp.confidence
        )
        return new_hyp

    def analyze_graph(self) -> List[Dict[str, Any]]:
        """
        Scans the knowledge graph for statistical patterns.
        """
        logger.info("[PI] Analyzing Knowledge Graph for emergent patterns...")
        return []
