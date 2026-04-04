import logging
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from core.lab_objects import ResearchIdea, Hypothesis, Experiment, Insight, Critique
from core.state import SurfaceState
from core.action import MutationAction, ActionType
from agents.base_agent import BaseAgent
from optimization.surrogate_models import SurrogateModel
from core.transition import TransitionEngine
from memory.storage_provider import StorageRegistry

logger = logging.getLogger(__name__)

class PostdocAgent(BaseAgent):
    """
    Agent: The Postdoc (Central Brain & Scientific Gatekeeper).
    
    The primary intellectual authority. Translates ResearchIdeas into 
    falsifiable Hypotheses and concrete Experiments.
    
    Authority:
    - Can reject or revise PI ideas.
    - Mandatory memory retrieval before any decision.
    - Owns the transformation logic: Idea -> Critique -> Hypothesis -> Experiment.
    """
    def __init__(self, surrogate: SurrogateModel, storage: StorageRegistry):
        super().__init__()
        self.belief_state = surrogate
        self.storage = storage
        self.transition_engine = TransitionEngine()

    def review_idea(self, idea: ResearchIdea) -> Critique:
        """
        Rule: Postdoc must validate PI ideas. 
        Mandatory memory check to see if we've tried this before.
        """
        logger.info(f"[Postdoc] Reviewing PI idea: {idea.goal}")
        
        # 1. Mandatory Memory Retrieval
        similar_past_results = self.storage.retrieve_similar_results(idea.intuition, top_k=3)
        
        validity = True
        issues = []
        revision = None
        
        # Heuristic critique logic (In production: LLM reasoning over memory)
        if "random" in idea.intuition.lower():
            validity = False
            issues.append("Combinatorial explosion; no physical basis for random search.")
            revision = "Focus on transition metal B-site substitutions (Co, Fe, Ni) with r_ionic similarity."
        
        # 2. Check memory for redundancies
        if any(res.get("reward", -1e9) > -0.1 for res in similar_past_results):
            validity = False
            issues.append("Redundancy: Similar high-performing configurations already identified in memory.")
            revision = "Explore the boundary of the known high-performing region via vacancy induction."

        return Critique(
            idea_id=idea.id,
            validity=validity,
            issues=issues,
            revised_plan=revision,
            confidence=0.85
        )

    def formulate_hypothesis(self, idea: ResearchIdea, critique: Critique) -> Hypothesis:
        """
        Rule: Formalize testable Hypothesis from Idea + Critique.
        Must define variable, manipulation, expected_effect, and falsification_condition.
        """
        logger.info(f"[Postdoc] Formulating strict hypothesis.")
        
        active_intuition = critique.revised_plan if not critique.validity else idea.intuition
        
        # In production: LLM generates this structured object
        hyp = Hypothesis(
            idea_id=idea.id,
            variable="Surface Cation Substitution (B-site)",
            manipulation="Increase Co concentration at surface by 25%",
            expected_effect="Reduction in Oxygen Vacancy Formation Energy (E_v) by >0.1 eV",
            metric="vacancy_formation_energy",
            falsification_condition="E_v remains constant or increases with Co concentration",
            confidence=0.7
        )
        return hyp

    def analyze_trends_from_memory(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        """
        Rule: Postdoc MUST analyze past results before designing experiments.
        This provides context on 'what worked' and 'what failed'.
        """
        logger.info(f"[Postdoc] Analyzing memory for trends related to: {hypothesis.variable}")
        
        # 1. Retrieve all past experiments on this variable
        past_data = self.storage.experiment_db.get_training_data()
        
        relevant_results = []
        for res in past_data:
            # Check if this state has the target element or defect
            # (Simplified check: look for 'Co' substitutions in action parameters)
            action_params = res.get("action", {}).get("parameters", {})
            if action_params.get("dopant") == "Co" or action_params.get("original_element") == "Co":
                relevant_results.append(res)

        if not relevant_results:
            return {"trend": "no prior data", "confidence": 0.0}

        # 2. Heuristic trend detection (In production: LLM + Data Science tools)
        rewards = [r.get("reward", -1e9) for r in relevant_results]
        trend = "improving" if rewards[-1] > rewards[0] else "plateau/declining"
        
        return {
            "trend": trend,
            "best_reward": max(rewards),
            "n_samples": len(relevant_results),
            "confidence": 0.9 if len(relevant_results) > 3 else 0.5
        }

    def design_experiments(self, hypothesis: Hypothesis, current_state: SurfaceState) -> List[Experiment]:
        """
        Rule: Design experiments by leveraging both the hypothesis and past trends.
        """
        # 1. Mandatory Memory Trend Analysis
        trends = self.analyze_trends_from_memory(hypothesis)
        logger.info(f"[Postdoc] Trends from memory: {trends['trend']} (Best Reward: {trends.get('best_reward')})")

        experiments = []
        # 2. Refine design based on trends
        # If past results were declining, the postdoc might suggest a different substitution
        dopants = ["Co", "Ni"] if trends["trend"] != "declining" else ["Fe", "Mn"]
        
        for dopant in dopants:
            experiments.append(Experiment(
                hypothesis_id=hypothesis.id,
                parameters={
                    "state": current_state,
                    "action": MutationAction(
                        action_type=ActionType.SUBSTITUTIONAL_DOPANT,
                        parameters={"dopant": dopant, "original_element": "Ti"}
                    )
                },
                method="DFT" if trends["confidence"] > 0.8 else "MLIP",
                expected_output=["total_energy", "vacancy_formation_energy"]
            ))
            
        return experiments

    def analyze_results(self, hypothesis: Hypothesis, results: List[Dict[str, Any]]) -> Insight:
        """
        Rule: Interpret results back into scientific conclusions.
        """
        logger.info(f"[Postdoc] Analyzing results for hypothesis: {hypothesis.id}")
        
        # 1. Check falsification condition
        # (Simplified logic)
        best_ev = min([r.get("observables", {}).get("vacancy_formation_energy", 1e9) for r in results])
        falsified = best_ev > 2.0 # Threshold for the example
        
        conclusion = "Hypothesis SUPPORTED." if not falsified else "Hypothesis FALSIFIED."
        if falsified:
            conclusion += f" Falsification condition met: {hypothesis.falsification_condition}"
            
        return Insight(
            hypothesis_id=hypothesis.id,
            conclusion=conclusion,
            confidence=0.9,
            data_summary={"best_ev": best_ev, "sample_count": len(results)}
        )

    # BaseAgent implementation (unused in formal flow but kept for interface)
    def observe_state(self) -> Any: return self.storage.experiment_db.get_training_data()
    def update_belief(self, observations: Any) -> None: self.belief_state.update(observations)
    def propose_actions(self) -> List[Any]: return []
    def score_actions(self, actions: List[Any]) -> List[float]: return []
    def execute_best(self, best_action: Any) -> Any: return None
    def update_memory(self, result: Any) -> None: pass
