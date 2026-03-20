import logging
import os
from typing import Dict, Any, Optional
import json
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

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
        """
        # Automatically load from .env file if it exists
        load_dotenv()
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in your environment or .env file.")
            
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.model_id = "gemini-2.0-flash" # Optimized for structured output
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise

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

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=f"User Goal: {prompt}",
                config={
                    "system_instruction": system_instruction,
                    "response_mime_type": "application/json"
                }
            )
            config: Dict[str, Any] = json.loads(response.text)
            
            # 2. Literature Cross-Check (Smell Fix: Grounding)
            from memory.literature_db import LiteratureDatabase
            lit_db = LiteratureDatabase()
            lit_db.load()
            
            # Extract keywords from the proposed bulk composition
            bulk = config.get("constraints", {}).get("bulk", {})
            keywords = list(bulk.keys())
            claims = lit_db.find_claims(keywords)
            
            if claims:
                logger.info(f"Found {len(claims)} relevant claims in LiteratureDB.")
                prior_text = "\n[PRIOR DOMAIN KNOWLEDGE]:\n" + "\n".join([f"- {c}" for c in claims[:3]])
                config["description"] = prior_text + "\n\n" + config.get("description", "")
            
            return config
        except Exception as e:
            logger.error(f"Campaign conceptualization failed: {e}.")
            raise
