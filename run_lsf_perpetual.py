import os
import logging
import time
from agents.collaborator_agent import LLMCollaborator
from core.campaign_manager import CampaignManager

# Professional unbuffered logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PerpetualLSF")

def main():
    prompt = "I want to discover the mechanism of Cr and S poisoning on La0.6Sr0.4Co0.2Fe0.8O3 (LSCF) (001) surfaces. Specifically, explore how surface Sr-segregation and oxygen vacancies facilitate the formation of Sr-CrO4 and Sr-SO4 like structures. Use CHGNet to rapidly screen configurations and identify non-obvious scientific trajectories."
    
    print("\n" + "="*60)
    print("   CLASDE DISCOVERY: LSCF Cr/S POISONING MECHANISM")
    print("="*60)
    
    # LIVE TEST SETTINGS
    overrides = {
        "compute": {"platform": "local", "mode": "chgnet"},
        "budget": {"max_evaluations": 20},
        "name": "LSCF_Poisoning_Discovery",
        "optimization": {"batch_size": 2} # Demonstrate parallel polling
    }
    
    print(f"\n[System] Launching 24/7 discovery loop. Results persist in SQLite.")
    print("[System] The loop will automatically refine hypotheses iteration-by-iteration.")
    
    try:
        campaign = CampaignManager.from_prompt(prompt, overrides=overrides)
        campaign.run()
    except KeyboardInterrupt:
        print("\n[System] Perpetual loop paused by user. Knowledge Graph preserved.")

if __name__ == "__main__":
    main()
