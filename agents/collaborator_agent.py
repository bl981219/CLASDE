import logging
import os
import google.generativeai as genai
from typing import Dict, Any, Optional
import json
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class LLMCollaborator:
    """
    The Strategic Collaborator Agent (Agent -1).
    
    This agent serves as the entry point for human-machine collaboration. It utilizes 
    Large Language Models (LLMs) to parse natural language research intents into 
    the formalized JSON schema required by the CLASDE engine.
    
    Responsibilities:
    1. Semantic parsing of high-level goals (e.g., "Find Sr segregation trends").
    2. Mapping chemical entities to stoichiometric vectors.
    3. Suggesting relevant environmental variables (T, p, Phi) for the campaign.
    4. Providing a scientific rationale for the proposed strategy.
    """
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the collaborator with an API key and configure the generative model.
        Falls back to a heuristic-based 'mock' mode if the API is unavailable.
        """
        # Automatically load from .env file if it exists
        load_dotenv()
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.use_mock = os.getenv("CLASDE_MOCK_LLM", "false").lower() == "true"
        
        if not self.use_mock:
            if not self.api_key:
                # Silently default to mock if no credentials found
                self.use_mock = True
            else:
                try:
                    genai.configure(api_key=self.api_key)
                    # We use gemini-pro for stable structured output in research tasks
                    self.model = genai.GenerativeModel('gemini-pro')
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini API ({e}). Using mock mode.")
                    self.use_mock = True

    def translate_goal_to_campaign(self, prompt: str) -> Dict[str, Any]:
        """
        Translates a natural language string into a structured Campaign configuration.
        
        Args:
            prompt: The user's research question or high-level goal.
            
        Returns:
            A dictionary containing the campaign configuration (objective, constraints, budget).
        """
        if self.use_mock:
            return self._mock_translation(prompt)

        # The system instruction establishes the 'persona' and constraints for the LLM
        system_instruction = """
        You are an expert computational surface scientist. Your task is to translate a 
        human researcher's high-level goal into a JSON configuration for the 
        CLASDE (Closed-Loop Autonomous Surface Discovery Engine).

        Constraints for JSON Output:
        - name: Short string identifier.
        - objective: { 'type': str, 'target_species': Optional[str], 'target_e_ads': Optional[float] }
          Types: 'adsorption_tuning', 'stability', 'segregation', 'functional'.
        - constraints: { 'bulk': Dict[str, float], 'facet': List[int, int, int] }
        - variables: List of strings (T, p, Phi).
        - budget: { 'max_evaluations': int }
        - description: Scientific summary of the objective.

        Example Input: "how does LaSrFeO3 Sr segregation depends on T, PO2, and overpotential"
        """

        try:
            # Generate the content with a JSON-enforcing response MIME type
            response = self.model.generate_content(
                f"{system_instruction}\n\nUser Goal: {prompt}\n\nJSON Output:",
                generation_config={"response_mime_type": "application/json"}
            )
            config: Dict[str, Any] = json.loads(response.text)
            return config
        except Exception as e:
            logger.error(f"LLM translation failed: {e}. Falling back to internal heuristics.")
            return self._mock_translation(prompt)

    def _mock_translation(self, prompt: str) -> Dict[str, Any]:
        """
        A rule-based heuristic fallback for demonstration and offline testing.
        Identifies key terms in the prompt to construct a reasonable campaign.
        """
        prompt_lower = prompt.lower()
        
        # Heuristic 1: Detect Sr Segregation in Perovskites
        if "segregation" in prompt_lower and "sr" in prompt_lower:
            return {
                "name": "Sr_Segregation_Study",
                "objective": {
                    "type": "segregation",
                    "target_species": "Sr"
                },
                "constraints": {
                    "bulk": {"La": 0.5, "Sr": 0.5, "Fe": 1.0, "O": 3.0},
                    "facet": [0, 0, 1]
                },
                "variables": ["T", "p", "Phi"],
                "budget": {"max_evaluations": 20},
                "description": "Rule-based mapping: Investigating Sr segregation at the LSF (001) surface."
            }
        
        # Heuristic 2: Detect Cu 111 and Oxygen
        if "cu" in prompt_lower and "111" in prompt_lower:
            return {
                "name": "Oxygen_on_Cu111",
                "objective": {
                    "type": "adsorption_tuning",
                    "adsorbate": "O",
                    "target_e_ads": -1.5
                },
                "constraints": {
                    "bulk": {"Cu": 1.0},
                    "facet": [1, 1, 1]
                },
                "budget": {"max_evaluations": 15},
                "description": "Rule-based mapping: Studying oxygen adsorption behavior on Cu(111)."
            }
        
        # Heuristic 3: General Stability Search
        return {
            "name": "General_Discovery",
            "objective": {"type": "stability"},
            "constraints": {
                "bulk": {"La": 0.5, "Sr": 0.5, "Mn": 1.0, "O": 3.0},
                "facet": [0, 0, 1]
            },
            "budget": {"max_evaluations": 10},
            "description": f"General stability search derived from: {prompt}"
        }
