from typing import Dict, Any, List
from core.campaign import Campaign
from core.action import ActionType

class ResearchPlanner:
    """
    Agent 0.5 — The Research Planner.
    
    Acts as the bridge between the high-level PI (Hypothesis Agent) and the execution 
    layer (Optimization Strategist). It converts abstract hypotheses into concrete, 
    executable `Campaign` objects with defined search spaces and budgets.
    """
    def __init__(self):
        pass

    def generate_campaigns(self, hypotheses: List[Dict[str, Any]]) -> List[Campaign]:
        """
        Convert a list of hypotheses into structured experimental campaigns.
        """
        campaigns = []
        for i, hyp in enumerate(hypotheses):
            feature = hyp.get("feature", "unknown_feature")
            effect = hyp.get("effect", "unknown_effect")
            
            # Formulate the objective based on the hypothesis
            # For demonstration, we assume we are testing an adsorption tuning hypothesis
            objective = {
                "type": "adsorption_tuning",
                "target_e_ads": -1.5,
                "adsorbate": "O"
            }
            
            # Map features to actions
            action_space = [ActionType.CHANGE_TERMINATION, ActionType.MODIFY_COVERAGE]
            if "vacancy" in feature.lower():
                action_space.append(ActionType.INTRODUCE_VACANCY)
            if "dopant" in feature.lower() or "substitution" in feature.lower():
                action_space.append(ActionType.SUBSTITUTIONAL_DOPANT)
                
            # Assume a diverse material space to test the hypothesis broadly
            material_space = ["LaFeO3", "SrTiO3", "BaCoO3"]
            
            campaign = Campaign(
                name=f"Test_Hypothesis_{i+1}",
                objective=objective,
                material_space=material_space,
                action_space=action_space,
                budget=50,
                description=f"Testing theory: {feature} -> {effect} with confidence {hyp.get('confidence', 0.0):.2f}"
            )
            campaigns.append(campaign)
            
        return campaigns
