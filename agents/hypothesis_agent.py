import logging
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
        self.hyp_db.add_hypothesis(hypothesis)
        return hypothesis

    def verify_current_hypothesis(self, current_hyp: ScientificHypothesis, dataset: List[Dict[str, Any]]) -> str:
        """
        Critical Review: Compares experimental results against the hypothesis.
        """
        if not dataset:
            return "No data to verify hypothesis."

        logger.info(f"[PI] Verifying Hypothesis: {current_hyp.theory_statement}")
        
        # Simple Logic: Check if rewards are improving or showing the predicted trend
        # Real logic would use the TheoryBuilder's correlations
        success_count = len([d for d in dataset if d.get("reward", -1e9) > -2.0])
        
        if success_count > 0:
            current_hyp.status = "verified"
            return f"Hypothesis SUPPORTED by {success_count} data points. Proceeding to refine theory."
        else:
            current_hyp.status = "falsified"
            return "Hypothesis NOT supported. Pivot required in next research cycle."

    def evolve_hypothesis(self, old_hyp: ScientificHypothesis, discoveries: List[Dict[str, Any]]) -> ScientificHypothesis:
        """Generates the next hypothesis based on the feedback loop."""
        new_statement = f"Refined theory: {old_hyp.theory_statement} with enhanced focus on vacancy stability."
        new_hyp = ScientificHypothesis(
            theory_statement=new_statement,
            target_property=old_hyp.target_property,
            predicted_trend="linear",
            status="untested"
        )
        self.hyp_db.add_hypothesis(new_hyp)
        return new_hyp
