import os
import argparse
import json
from dotenv import load_dotenv
from agents.collaborator_agent import LLMCollaborator
from core.campaign_manager import CampaignManager

def main():
    # Ensure .env credentials are available when running from project root.
    load_dotenv()

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

    # --- Act as Research Governor (Lab Manager) ---
    if os.getenv("CLASDE_AUTO_CONFIRM", "false").lower() == "true":
        choice = "p"
    else:
        print("\n[Lab Manager Action Required]")
        print("Do you want to (P)roceed, (M)odify constraints, or (A)bort?")
        try:
            choice = input("Choice [P/m/a]: ").strip().lower()
            if choice == "": choice = "p"
        except EOFError:
            choice = "a"

    if choice == "m":
        # Manual Budget Override
        current_budget = config.get("budget", {}).get("max_evaluations", req_budget)
        new_budget = input(f"Set max evaluations (current: {current_budget}): ").strip()
        if new_budget.isdigit():
            config["budget"] = {"max_evaluations": int(new_budget)}
        
        # Manual Facet Override
        current_facet = config.get("constraints", {}).get("facet", [0,0,1])
        new_facet = input(f"Set Miller index facet (current: {current_facet}): ").strip()
        try:
            if new_facet: config["constraints"]["facet"] = eval(new_facet)
        except:
            print("Invalid facet format. Keeping default.")
        
        choice = "p" # Proceed with updated config

    if choice == "p":
        print("\n[System] Handing over to Research Governor. Starting loop...")
        
        # Merge defaults and include the original prompt for documentation
        config["original_prompt"] = user_prompt
        
        if "acquisition" not in config:
            config["acquisition"] = {"acquisition_type": "EI", "kappa": 2.576}
        if "compute" not in config:
            config["compute"] = {"mode": "local_emt"}

        campaign = CampaignManager(config)
        campaign.run()
    else:
        print("\n[System] Campaign aborted by user.")

if __name__ == "__main__":
    main()
