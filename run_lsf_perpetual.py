import os
import logging
import time
from agents.collaborator_agent import LLMCollaborator
from workflows.adsorption_workflow import run_adsorption_campaign

# Professional unbuffered logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PerpetualLSF")

def main():
    prompt = "I am interested in how Sr-segregation behaves on La0.5Sr0.5FeO3 (LSF) (001) surfaces. Please perform a research campaign where you use the universal CHGNet potential to explore a range of surface cation configurations."
    
    print("\n" + "="*60)
    print("   CLASDE PERPETUAL DISCOVERY MODE: LSF SEGREGATION")
    print("="*60)
    
    collaborator = LLMCollaborator()
    config = collaborator.translate_goal_to_campaign(prompt)
    
    # PERPETUAL SETTINGS
    config["compute"] = {"platform": "local", "mode": "chgnet"}
    config["budget"] = {"max_evaluations": 1000} # Near-infinite for 24/7
    config["name"] = "LSF_Perpetual_Segregation_Study"
    config["original_prompt"] = prompt
    
    print(f"\n[System] Launching 24/7 discovery loop. Results persist in SQLite.")
    print("[System] The loop will automatically refine hypotheses iteration-by-iteration.")
    
    try:
        run_adsorption_campaign(config)
    except KeyboardInterrupt:
        print("\n[System] Perpetual loop paused by user. Knowledge Graph preserved.")

if __name__ == "__main__":
    main()
