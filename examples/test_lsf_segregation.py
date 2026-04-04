import logging
import sys
import os

# Add the project root to the path so we can import core/agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.campaign_manager import CampaignManager
from agents.collaborator_agent import LLMCollaborator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_lsf_test():
    prompt = "I am interested in how Sr-segregation behaves on La0.5Sr0.5FeO3 (LSF) (001) surfaces. Please perform a research campaign where you use the universal CHGNet potential to explore a range of surface cation configurations."
    
    logger.info(f"Starting LSF Segregation Test with prompt: {prompt}")
    
    # 1. Bootstrap config from prompt using the Collaborator
    collaborator = LLMCollaborator()
    config = collaborator.translate_goal_to_campaign(prompt)
    config["original_prompt"] = prompt
    config["results_dir"] = "examples/lsf_segregation_results"
    
    # Ensure compute is set to local/ase for CHGNet
    config["compute"] = {
        "platform": "local",
        "mode": "chgnet"
    }
    
    # Set a small budget for this test
    config["budget"] = {"max_evaluations": 5}
    
    # 2. Initialize CampaignManager
    manager = CampaignManager(config)
    
    # 3. Run the campaign
    logger.info("--- Executing Campaign ---")
    manager.run()
    logger.info("--- Campaign Complete ---")
    
    logger.info(f"Results saved in {config['results_dir']}")

if __name__ == "__main__":
    run_lsf_test()
