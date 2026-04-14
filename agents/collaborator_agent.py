import logging
import os
from typing import Dict, Any, Optional
import json
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class LLMCollaborator:
    """
    The Strategic Collaborator Agent (Agent -1).
    
    This agent serves as the entry point for human-machine collaboration. It utilizes 
    Large Language Models (LLMs) to parse natural language research intents into 
    the formalized JSON schema required by the CLASDE engine.
    """
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the collaborator with an API key and configure the generative model.
        Falls back to a heuristic-based 'mock' mode if the API is unavailable.
        """
        # Automatically load from .env file if it exists
        load_dotenv()
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.model_id = "gemini-2.0-flash" # Optimized for structured output
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini API: {e}. Using mock mode.")
            self.api_key = None

    def translate_goal_to_campaign(self, prompt: str) -> Dict[str, Any]:
        """
        Translates a natural language string into a structured Campaign configuration.
        Includes a LiteratureDatabase cross-check to ground the goal in prior knowledge.
        """
        # 1. LLM Conceptualization
        system_instruction = """
        You are an expert computational surface scientist. Your task is to translate a 
        human researcher's high-level goal into a JSON configuration for the 
        CLASDE (Closed-Loop Autonomous Surface Discovery Engine).

        Constraints for JSON Output:
        - name: Short string identifier.
        - objective: { 'type': str, 'target_species': Optional[str], 'target_e_ads': Optional[float] }
          Types: 'adsorption_tuning', 'stability', 'functional'.
        - constraints: { 'bulk': Dict[str, float], 'facet': List[int, int, int] }
        - variables: List of strings (T, p, Phi).
        - budget: { 'max_evaluations': int }
        - description: Scientific summary of the objective.
        """

        config = None
        if self.api_key:
            try:
                config = self._call_llm(prompt, system_instruction)
            except Exception as e:
                logger.warning(
                    "LLM translation failed after retries: %s. Falling back to internal heuristics.", e
                )

        if not config:
            config = self._mock_translation(prompt)
            
        # 2. Optional Literature Cross-Check
        if config.get("literature_check", False):
            from memory.literature_db import LiteratureDatabase
            lit_db = LiteratureDatabase()
            lit_db.load()
            
            bulk = config.get("constraints", {}).get("bulk", {})
            keywords = list(bulk.keys())
            claims = lit_db.find_claims(keywords)
            
            if claims:
                logger.info(f"Found {len(claims)} relevant claims in LiteratureDB.")
                prior_text = "\n[PRIOR DOMAIN KNOWLEDGE]:\n" + "\n".join([f"- {c}" for c in claims[:3]])
                config["description"] = prior_text + "\n\n" + config.get("description", "")
        
        return config

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def _call_llm(self, prompt: str, system_instruction: str) -> Dict[str, Any]:
        """Makes the LLM API call with automatic retries on transient failures."""
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=f"User Goal: {prompt}",
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json"
            }
        )
        return json.loads(response.text)

    def _mock_translation(self, prompt: str) -> Dict[str, Any]:
        """
        A rule-based heuristic fallback for demonstration and offline testing.
        """
        prompt_lower = prompt.lower()
        
        # Heuristic 1: Detect Sr Segregation in Perovskites
        if "segregation" in prompt_lower and "sr" in prompt_lower:
            return {
                "name": "Sr_Segregation_Study",
                "objective": {
                    "type": "stability"
                },
                "constraints": {
                    "bulk": {"La": 0.5, "Sr": 0.5, "Fe": 1.0, "O": 3.0},
                    "facet": [0, 0, 1]
                },
                "variables": ["T", "p", "Phi"],
                "budget": {"max_evaluations": 20},
                "description": "Rule-based mapping: Investigating Sr segregation at the LSF (001) surface via stability minimization."
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
        
        # Heuristic 3: Detect LSCF Poisoning (Cr and S)
        if "lscf" in prompt_lower or ("la" in prompt_lower and "sr" in prompt_lower and "fe" in prompt_lower):
            return {
                "name": "LSCF_Poisoning_Mechanism",
                "objective": {
                    "type": "adsorption_tuning",
                    "adsorbate": "SO2",
                    "target_e_ads": -1.5
                },
                "constraints": {
                    "bulk": {"La": 0.6, "Sr": 0.4, "Fe": 0.8, "Co": 0.2, "O": 3.0},
                    "facet": [0, 0, 1]
                },
                "variables": ["T", "p", "Phi"],
                "budget": {"max_evaluations": 100},
                "description": "Investigating the competition between CrO3 and SO2 poisoning on La0.6Sr0.4Fe0.8Co0.2O3 (001) surfaces."
            }

        # Heuristic 4: STO B-site Doping
        if ("sto" in prompt_lower or "srtio3" in prompt_lower) and "doping" in prompt_lower:
            return {
                "name": "STO_B_Site_Doping_ORR",
                "objective": {
                    "type": "adsorption_tuning",
                    "adsorbate": "O",
                    "target_e_ads": -1.2
                },
                "constraints": {
                    "bulk": {"Sr": 1.0, "Ti": 1.0, "O": 3.0},
                    "facet": [0, 0, 1]
                },
                "budget": {"max_evaluations": 20},
                "description": "Optimizing B-site dopants on SrTiO3 (001) for ORR activity."
            }
        
        # Heuristic 5: General Stability Search
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
