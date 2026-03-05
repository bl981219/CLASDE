import os
import yaml
import argparse
from workflows.adsorption_campaign import run_adsorption_campaign

def main():
    parser = argparse.ArgumentParser(description="CLASDE: Closed-Loop Atomistic Surface Design Engine")
    parser.add_argument("--config", type=str, default="campaign.yaml", help="Path to campaign YAML configuration.")
    args = parser.parse_args()

    # 0. Load Campaign Configuration
    if os.path.exists(args.config):
        print(f"Loading campaign from {args.config}...")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Fallback to defaults if no config found
        print(f"Warning: {args.config} not found. Using default internal configuration.")
        config = {
            "objective": {
                "type": "adsorption_tuning", 
                "target_e_ads": -1.5
            },
            "budget": {
                "max_evaluations": 6
            },
            "acquisition": {
                "acquisition_type": "EI", 
                "kappa": 2.576
            },
            "constraints": {
                "facet": [0, 0, 1],
                "bulk": {"La": 0.5, "Sr": 0.5, "Mn": 1.0, "O": 3.0}
            }
        }
    
    run_adsorption_campaign(config)

if __name__ == "__main__":
    main()
