import logging
import os
import sys

# Allow running as: python3 examples/test_surface_segregation_lsf_chgnet.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.collaborator_agent import LLMCollaborator
from core.campaign_manager import CampaignManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


PROMPT = (
    "I am interested in how Sr-segregation behaves on La0.5Sr0.5FeO3 (LSF) (001) "
    "surfaces. Please perform a research campaign where you use the universal "
    "CHGNet potential to explore a range of surface cation configurations."
)


def run_surface_segregation_lsf_chgnet_test() -> str:
    collaborator = LLMCollaborator()
    config = collaborator.translate_goal_to_campaign(PROMPT)
    config["original_prompt"] = PROMPT
    config["results_dir"] = "examples/surface_segregation_lsf_chgnet_results"

    # Keep this as a fast, reproducible screening test.
    config["compute"] = {
        "platform": "local",
        "mode": "chgnet",
    }
    config["budget"] = {"max_evaluations": 1}

    logger.info("Starting LSF Sr-segregation CHGNet screening test.")
    manager = CampaignManager(config)
    manager.run()
    logger.info("Campaign finished. Results: %s", config["results_dir"])
    return config["results_dir"]


if __name__ == "__main__":
    out_dir = run_surface_segregation_lsf_chgnet_test()
    print(f"Saved test results to: {out_dir}")
