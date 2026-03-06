import os
import argparse
import json
from agents.collaborator import LLMCollaborator
from workflows.autonomous_campaign import run_adsorption_campaign

def main():
    parser = argparse.ArgumentParser(description="CLASDE Collaborator: Natural Language Research Interface")
    parser.add_argument("--prompt", type=str, help="Your research question or goal.")
    parser.add_argument("--key", type=str, help="Google API Key (optional, or use GOOGLE_API_KEY env var).")
    args = parser.parse_args()

    api_key = args.key or os.getenv("GOOGLE_API_KEY")
    is_mock = os.getenv("CLASDE_MOCK_LLM", "false").lower() == "true"
    
    if not api_key and not is_mock:
        print("Error: No Google API Key found. Provide via --key, set GOOGLE_API_KEY environment variable, or use CLASDE_MOCK_LLM=true.")
        return

    collaborator = LLMCollaborator(api_key=api_key)

    if args.prompt:
        user_prompt = args.prompt
    else:
        print("\n" + "="*50)
        print("   CLASDE COLLABORATOR: NATURAL LANGUAGE INTERFACE")
        print("="*50)
        user_prompt = input("\nWhat is your research goal today? (e.g., 'how does Sr segregate in LaSrFeO3?')\n> ")

    print(f"\n[Collaborator Agent] Analyzing request and formulating strategy...")
    config = collaborator.translate_goal_to_campaign(user_prompt)

    if not config:
        print("Failed to generate campaign configuration. Please try a more specific prompt.")
        return

    print("\n" + "-"*50)
    print("   PROPOSED AUTONOMOUS RESEARCH CAMPAIGN")
    print("-"*50)
    print(f"  Name:        {config.get('name')}")
    print(f"  Description: {config.get('description')}")
    print(f"  Objective:   {config.get('objective')}")
    print(f"  Chemistry:   {config.get('constraints', {}).get('bulk')}")
    print(f"  Facet:       {config.get('constraints', {}).get('facet')}")
    print(f"  Variables:   {config.get('variables')}")
    
    # Show requested budget vs actual (Governor will enforce the ceiling)
    req_budget = config.get("budget", {}).get("max_evaluations", 20)
    print(f"  Budget:      {req_budget} iterations")
    print("-"*50)

    # Sanity Check & Confirmation
    if os.getenv("CLASDE_AUTO_CONFIRM", "false").lower() == "true":
        confirm = "yes"
    else:
        try:
            print("\nCRITICAL: Starting this campaign will initiate autonomous structure")
            print("generation and computational evaluations (DFT/MLFF).")
            confirm = input("\nDo you want to proceed with this campaign? [Y/n]: ").strip().lower()
            if confirm == "": confirm = "y" # Default to yes
        except EOFError:
            confirm = "no"
            print("\nNon-interactive mode detected. Use CLASDE_AUTO_CONFIRM=true to skip confirmation.")

    if confirm in ["y", "yes"]:
        print("\n[System] Handing over to Research Governor. Starting loop...")
        
        # Merge defaults and include the original prompt for documentation
        config["budget"] = config.get("budget", {"max_evaluations": req_budget})
        config["original_prompt"] = user_prompt
        
        if "acquisition" not in config:
            config["acquisition"] = {"acquisition_type": "EI", "kappa": 2.576}
        if "compute" not in config:
            config["compute"] = {"mode": "local_emt"}

        run_adsorption_campaign(config)
    else:
        print("\n[System] Campaign aborted by user.")

if __name__ == "__main__":
    main()
