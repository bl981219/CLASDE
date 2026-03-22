import os
import logging
from agents.collaborator_agent import LLMCollaborator
from core.campaign_manager import CampaignManager

# Professional unbuffered logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("STO_Doping")

def main():
    prompt = "I want to optimize the oxygen reduction reaction (ORR) activity on SrTiO3 (STO) by doping the B-site with transition metals (Ta, Mn, Fe, Co, Ni). Run a closed-loop tuning campaign using CHGNet to rapidly screen different dopant types and concentrations at the surface. Your goal is to find a dopant configuration that brings the Oxygen adsorption energy (E_ads) closest to -1.2 eV."
    
    print("\n" + "="*60)
    print("   CLASDE DISCOVERY: STO B-SITE DOPING OPTIMIZATION")
    print("="*60)
    
    # Run the campaign with overrides
    overrides = {
        "compute": {"platform": "local", "mode": "chgnet"},
        "budget": {"max_evaluations": 10},
        "name": "STO_B_Site_Doping_Study",
        "optimization": {"batch_size": 2} # Demonstrate parallel polling
    }
    
    campaign = CampaignManager.from_prompt(prompt, overrides=overrides)
    campaign.run()

if __name__ == "__main__":
    main()
