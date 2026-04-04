import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from core.lab_objects import ResearchIdea, Hypothesis, Experiment, Insight, Critique
from core.state import SurfaceState
from core.action import MutationAction, ActionType
from agents.base_agent import BaseAgent
from optimization.surrogate_models import SurrogateModel
from core.transition import TransitionEngine

logger = logging.getLogger(__name__)

class PostdocAgent(BaseAgent):
    """
    Agent: The Postdoc (Senior Knowledge Transformer).
    
    The central intelligence of the lab. Translates high-level Theory (PI) 
    into testable Hypotheses and concrete Experiments.
    
    Responsibilities:
    1. Critique PI ideas for physical validity and combinatorial sanity.
    2. Formalize valid ideas into testable Hypotheses.
    3. Design Experiments (simulations) to test Hypotheses.
    4. Interpret Results into Insights.
    5. Manage Memory (remembering failed trends).
    """
    def __init__(self, surrogate: SurrogateModel, experiment_db: Any, knowledge_graph: Any):
        super().__init__()
        self.belief_state = surrogate
        self.experiment_db = experiment_db
        self.kg = knowledge_graph
        self.transition_engine = TransitionEngine()

    def critique_idea(self, idea: ResearchIdea) -> Critique:
        """Rule 2: Postdoc must validate PI ideas. Debate step."""
        logger.info(f"[Postdoc] Critiquing PI's idea: {idea.goal}")
        
        # Heuristic critique (In production, this would be an LLM)
        validity = True
        issues = []
        revision = ""
        
        if "random" in idea.intuition.lower():
            validity = False
            issues.append("Combinatorial explosion from random doping.")
            revision = "Focus on transition metals with similar ionic radii to maintain perovskite structure."
        
        return Critique(
            idea_id=idea.id,
            validity=validity,
            issues=issues,
            suggested_revision=revision,
            confidence=0.8
        )

    def formulate_hypothesis(self, idea: ResearchIdea) -> Hypothesis:
        """Rule 3: Hypothesis must be testable (measurable metric, controllable variable)."""
        logger.info(f"[Postdoc] Translating idea into testable hypothesis.")
        
        # Example: Translating 'improve stability' idea into concrete hypothesis
        hyp = Hypothesis(
            idea_id=idea.id,
            variable="surface cation arrangement",
            change="segregation of larger cations",
            expected_effect="decrease surface grand potential",
            metric="stability",
            test_plan={"sampling_strategy": "active_learning"}
        )
        return hyp

    def design_experiments(self, hypothesis: Hypothesis, current_state: SurfaceState) -> List[Experiment]:
        """Rule 4: Hypothesis must generate concrete experiments."""
        logger.info(f"[Postdoc] Designing experiments for hypothesis: {hypothesis.id}")
        
        # Integration with Bayesian Optimization (Surrogate Model)
        # This replaces the old 'OptimizationStrategist.propose_actions'
        experiments = []
        
        # Example: Generate 3 candidate mutations based on the hypothesis
        for dopant in ["Sr", "Ba", "Ca"]:
            experiments.append(Experiment(
                hypothesis_id=hypothesis.id,
                parameters={
                    "state": current_state,
                    "action": MutationAction(
                        action_type=ActionType.SUBSTITUTIONAL_DOPANT,
                        parameters={"dopant": dopant, "original_element": "La"}
                    )
                },
                method="MLIP", # Default to cheap fidelity first
                expected_output=["total_energy", "structure"]
            ))
            
        return experiments

    def interpret_results(self, hypothesis: Hypothesis, experiments: List[Experiment], results: List[Dict[str, Any]]) -> Insight:
        """Translates Level 3 (Experiments) back to Level 4 (Insights)."""
        logger.info(f"[Postdoc] Analyzing results to generate insight.")
        
        rewards = [r.get("reward", -1e9) for r in results]
        max_reward = max(rewards) if rewards else -1e9
        
        conclusion = "Hypothesis partially supported." if max_reward > 0.0 else "Hypothesis falsified."
        
        return Insight(
            hypothesis_id=hypothesis.id,
            experiment_ids=[e.id for e in experiments],
            conclusion=conclusion,
            confidence=0.7,
            data_summary={"best_reward": max_reward}
        )

    # BaseAgent Abstract Methods (Required for loop)
    def observe_state(self) -> Any: return self.experiment_db.get_training_data()
    def update_belief(self, observations: Any) -> None: self.belief_state.update(observations)
    def propose_actions(self) -> List[Any]: return [] # Not used in this formal flow
    def score_actions(self, actions: List[Any]) -> List[float]: return []
    def execute_best(self, best_action: Any) -> Any: return None
    def update_memory(self, result: Any) -> None: pass
