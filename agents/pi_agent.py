import logging
from typing import Dict, Any, Optional
from core.lab_objects import ResearchIdea
from agents.collaborator_agent import LLMCollaborator

logger = logging.getLogger(__name__)

class PIAgent:
    """
    Agent: Principal Investigator (PI).
    
    Responsible for formulating high-level research goals and scientific intuition.
    The PI provides the 'What' and 'Why', but not the 'How'.
    """
    def __init__(self, collaborator: Optional[LLMCollaborator] = None):
        self.collaborator = collaborator or LLMCollaborator()

    def propose_idea(self, human_goal: str) -> ResearchIdea:
        """Translates natural language human intent into a formal ResearchIdea."""
        logger.info(f"[PI] Formulating research idea from goal: {human_goal}")
        
        # In a real system, use LLM to extract intuition and constraints
        # Fallback to structured parsing if collaborator fails
        config = self.collaborator.translate_goal_to_campaign(human_goal)
        
        idea = ResearchIdea(
            goal=config.get("name", "Unknown Campaign"),
            intuition=config.get("description", human_goal),
            constraints=config.get("constraints", {}),
            source="LLM-assisted human"
        )
        return idea

    def refine_idea(self, idea: ResearchIdea, feedback: str) -> ResearchIdea:
        """Updates the idea based on Postdoc critique."""
        logger.info(f"[PI] Refining idea based on feedback: {feedback}")
        idea.intuition += f"\n[Refinement]: {feedback}"
        return idea
