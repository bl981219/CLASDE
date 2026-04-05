import logging
import uuid
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential
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
    """
    def __init__(self, surrogate: SurrogateModel, storage: StorageRegistry, api_key: Optional[str] = None):
        super().__init__()
        self.belief_state = surrogate
        self.storage = storage
        self.transition_engine = TransitionEngine()
        
        load_dotenv()
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.model_id = "gemini-2.0-flash"
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini API in PostdocAgent: {e}. Using mock mode.")
            self.api_key = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def _call_llm(self, system_instruction: str, prompt: str) -> Dict[str, Any]:
        """Helper to call LLM with retry and structured output."""
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json"
            }
        )
        return json.loads(response.text)

    def review_idea(self, idea: ResearchIdea) -> Critique:
        """Gatekeeping: Validates PI ideas against physical laws and memory."""
        logger.info(f"[Postdoc] Reviewing PI idea: {idea.goal}")
        past_results = self.storage.retrieve_similar_results(idea.intuition, top_k=5)
        memory_context = json.dumps([{"reward": r.get("reward"), "config": str(r.get("state"))} for r in past_results])

        if not self.api_key:
            return self._mock_review(idea, past_results)

        system_instruction = """
        You are a senior Postdoc in a surface science lab. Your job is to CRITIQUE the PI's research idea.
        Output JSON: { "validity": bool, "issues": [str], "revised_plan": str, "confidence": float }
        """
        prompt = f"PI Idea: {idea.goal}\nIntuition: {idea.intuition}\nPast Memory: {memory_context}"

        try:
            data = self._call_llm(system_instruction, prompt)
            return Critique(idea_id=idea.id, **data)
        except Exception as e:
            logger.error(f"LLM Review failed after retries: {e}. Falling back to mock.")
            return self._mock_review(idea, past_results)

    def formulate_hypothesis(self, idea: ResearchIdea, critique: Critique) -> Hypothesis:
        """Transformation: Idea -> Falsifiable Hypothesis."""
        logger.info(f"[Postdoc] Formulating falsifiable hypothesis.")
        
        if not self.api_key:
            return self._mock_formulate(idea, critique)

        system_instruction = """
        You are a senior Postdoc. Convert a research idea and its critique into a formal, falsifiable hypothesis.
        Output JSON: { "variable": str, "manipulation": str, "expected_effect": str, "metric": str, "falsification_condition": str, "confidence": float }
        Metric Allowlist: [E_ads, stability, reward, adsorption_energy, vacancy_formation_energy, work_function]
        """
        prompt = f"Idea: {idea.intuition}\nCritique: {critique.revised_plan or 'Idea is valid.'}"

        try:
            data = self._call_llm(system_instruction, prompt)
            return Hypothesis(idea_id=idea.id, **data)
        except Exception as e:
            logger.error(f"LLM Hypothesis failed after retries: {e}. Falling back to mock.")
            return self._mock_formulate(idea, critique)

    def analyze_trends_from_memory(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        """Mandatory memory reasoning step."""
        logger.info(f"[Postdoc] Analyzing memory for trends related to: {hypothesis.variable}")
        past_data = self.storage.experiment_db.get_training_data()
        relevant = [r for r in past_data if hypothesis.variable.lower() in str(r.get("action")).lower()]
        
        if not relevant:
            return {"trend": "neutral", "confidence": 0.0}
            
        rewards = [r.get("reward", -1e9) for r in relevant]
        trend = "improving" if len(rewards) > 1 and rewards[-1] > rewards[0] else "plateau"
        return {"trend": trend, "best": max(rewards), "count": len(relevant)}

    def design_experiments(self, hypothesis: Hypothesis, current_state: SurfaceState) -> List[Experiment]:
        """Planning: Strategy -> Experiment. Uses LLM to propose candidate actions."""
        logger.info(f"[Postdoc] Designing experiments for hypothesis: {hypothesis.id}")
        
        if not self.api_key:
            return self._mock_design(hypothesis, current_state)

        system_instruction = """
        You are a senior Postdoc. Propose 3 candidate experimental actions to test the given hypothesis.
        Each action must be a mutation of the current surface state.
        
        Action Types: ['substitutional_dopant', 'vacancy_induction', 'adsorbate_placement']
        
        Output JSON: {
            "proposals": [
                {
                    "action_type": str,
                    "parameters": dict,
                    "rationale": str,
                    "fidelity": "DFT" | "MLIP"
                }
            ]
        }
        """
        prompt = f"Hypothesis: {hypothesis.model_dump_json()}\nCurrent State: {current_state.get_summary()}"

        try:
            data = self._call_llm(system_instruction, prompt)
            experiments = []
            for prop in data.get("proposals", []):
                experiments.append(Experiment(
                    hypothesis_id=hypothesis.id,
                    parameters={
                        "state": current_state,
                        "action": MutationAction(
                            action_type=ActionType(prop["action_type"]),
                            parameters=prop["parameters"]
                        ),
                        "rationale": prop.get("rationale")
                    },
                    method=prop.get("fidelity", "MLIP"),
                    expected_output=["total_energy", hypothesis.metric]
                ))
            return experiments[:3]
        except Exception as e:
            logger.error(f"LLM Experiment Design failed after retries: {e}. Falling back to mock.")
            return self._mock_design(hypothesis, current_state)

    def analyze_results(self, hypothesis: Hypothesis, results: List[Dict[str, Any]]) -> Insight:
        """Synthesis: Result -> Scientific Insight."""
        logger.info(f"[Postdoc] Analyzing results.")
        
        if not self.api_key:
            return self._mock_analyze(hypothesis, results)

        system_instruction = """
        You are a senior Postdoc. Analyze the experimental results against the original hypothesis.
        Determine if the hypothesis is supported or falsified.
        Output JSON: { "conclusion": str, "confidence": float, "data_summary": dict }
        """
        prompt = f"Hypothesis: {hypothesis.model_dump_json()}\nResults: {json.dumps(results, default=str)}"

        try:
            data = self._call_llm(system_instruction, prompt)
            return Insight(hypothesis_id=hypothesis.id, **data)
        except Exception as e:
            logger.error(f"LLM Analysis failed after retries: {e}. Falling back to mock.")
            return self._mock_analyze(hypothesis, results)

    # --- Mock Methods ---
    def _mock_review(self, idea: ResearchIdea, past: List) -> Critique:
        validity = True
        issues = []
        revision = None
        if "random" in idea.intuition.lower():
            validity = False
            issues.append("Heuristic: Random search detected in mock mode.")
            revision = "Focus on B-site substitutions."
        return Critique(idea_id=idea.id, validity=validity, issues=issues, revised_plan=revision, confidence=0.5)

    def _mock_formulate(self, idea: ResearchIdea, critique: Critique) -> Hypothesis:
        return Hypothesis(
            idea_id=idea.id,
            variable="dopant", manipulation="increase Co", 
            expected_effect="stability", metric="stability",
            falsification_condition="energy increases", confidence=0.5
        )

    def _mock_design(self, hypothesis: Hypothesis, current_state: SurfaceState) -> List[Experiment]:
        exp = Experiment(
            hypothesis_id=hypothesis.id,
            parameters={
                "state": current_state,
                "action": MutationAction(
                    action_type=ActionType.SUBSTITUTIONAL_DOPANT,
                    parameters={"dopant": "Co", "original_element": "Fe"}
                )
            },
            method="MLIP",
            expected_output=["total_energy", hypothesis.metric]
        )
        return [exp]

    def _mock_analyze(self, hypothesis: Hypothesis, results: List) -> Insight:
        return Insight(hypothesis_id=hypothesis.id, conclusion="Mock analysis completed.", confidence=0.5)

    # BaseAgent implementation
    def observe_state(self) -> Any: return self.storage.experiment_db.get_training_data()
    def update_belief(self, observations: Any) -> None: self.belief_state.update(observations)
    def propose_actions(self) -> List[Any]: return []
    def score_actions(self, actions: List[Any]) -> List[float]: return []
    def execute_best(self, best_action: Any) -> Any: return None
    def update_memory(self, result: Any) -> None: pass
