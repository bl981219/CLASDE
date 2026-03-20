import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from science.experiment_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

class ScientificHypothesis(BaseModel):
    """A formal scientific theory being tested by the engine."""
    theory_statement: str
    target_property: str
    predicted_trend: str # e.g., "Increasing" or "Decreasing"
    supporting_literature: List[str] = Field(default_factory=list)
    status: str = "untested" # untested, verified, falsified, inconclusive

class HypothesisAgent:
    """
    Agent 1 — The Principal Investigator (PI).
    
    The creative and critical lead. Formulates hypotheses from literature 
    and determines if they are supported by empirical data.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph, hypothesis_db: Any):
        self.kg = knowledge_graph
        self.hyp_db = hypothesis_db

    def formulate_initial_hypothesis(self, literature_claims: List[str], research_goal: str) -> ScientificHypothesis:
        """
        Synthesizes prior knowledge and user intent into a testable hypothesis.
        """
        logger.info("[PI] Reviewing literature and formulating initial hypothesis...")
        
        # Logic: In production, this uses the LLM to reason over claims.
        # Placeholder: Generate a theory based on the most common trend in claims.
        theory = f"Based on literature, the {research_goal} will be determined by surface cation arrangement."
        
        hypothesis = ScientificHypothesis(
            theory_statement=theory,
            target_property="adsorption_energy",
            predicted_trend="nonlinear",
            supporting_literature=literature_claims
        )
        self.hyp_db.add_hypothesis(
            hypothesis=hypothesis.theory_statement,
            evidence_ids=[], # Initial hypothesis has no empirical evidence yet
            confidence=0.5   # Baseline confidence
        )
        return hypothesis

    def verify_current_hypothesis(self, current_hyp: ScientificHypothesis, dataset: List[Dict[str, Any]]) -> str:
        """
        Critical Review: Compares experimental results against the hypothesis.
        Analyzes energy trends and reward optimization.
        """
        if len(dataset) < 2:
            return "Insufficient data for formal verification."

        logger.info(f"[PI] Verifying Hypothesis: {current_hyp.theory_statement}")
        
        # 1. Check for improvement trend
        rewards = [d.get("reward", -1e9) for d in dataset if d.get("reward") is not None]
        if not rewards: return "No rewards found in dataset."
        
        initial_reward = rewards[0]
        max_reward = max(rewards)
        improvement = max_reward - initial_reward
        
        # 2. Heuristic verification
        if improvement > 0.1: # Significant optimization found
            current_hyp.status = "verified"
            return f"Hypothesis SUPPORTED. Found configuration with reward {max_reward:.4f} (Improvement: {improvement:.4f} eV)."
        elif len(dataset) > 5:
            current_hyp.status = "falsified"
            return "Hypothesis NOT supported after 5+ iterations. Redefining search space."
        else:
            current_hyp.status = "untested"
            return "Initial results inconclusive. Continuing exploration."

    def evolve_hypothesis(self, old_hyp: ScientificHypothesis, dataset: List[Dict[str, Any]]) -> ScientificHypothesis:
        """Generates the next hypothesis based on the feedback loop discovery."""
        rewards = [d.get("reward", -1e9) for d in dataset]
        best_idx = int(np.argmax(rewards))
        best_state = dataset[best_idx]['state']
        
        # Extract features of the best configuration
        comp_summary = "".join([f"{k}{v}" for k, v in best_state.bulk_composition.items()])
        
        new_statement = f"The {comp_summary} arrangement has demonstrated local stability. Next, we test if Oxygen vacancy concentration further stabilizes this specific cation distribution."
        
        new_hyp = ScientificHypothesis(
            theory_statement=new_statement,
            target_property=old_hyp.target_property,
            predicted_trend="increasing stability with vacancy count",
            status="untested"
        )
        
        self.hyp_db.add_hypothesis(
            hypothesis=new_hyp.theory_statement,
            evidence_ids=[best_state.get_id()], 
            confidence=0.7 # High confidence in the local lead
        )
        return new_hyp

    def analyze_graph(self) -> List[Dict[str, Any]]:
        """
        Scans the knowledge graph for statistical patterns.
        Used by the TheoryBuilder to finalize discoveries.
        """
        logger.info("[PI] Analyzing Knowledge Graph for emergent patterns...")
        # Placeholder: In a full system, this performs graph queries for correlations
        return []
