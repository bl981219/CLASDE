import argparse
import logging
from execution.workflow_runner import run_adsorption_campaign

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="CLASDE Domain CLI: Explore Surface")
    parser.add_argument("material", type=str, help="Bulk material composition (e.g., LaSrFeO3)")
    parser.add_argument("facet", type=str, help="Miller index (e.g., 001)")
    parser.add_argument("adsorbate", type=str, help="Adsorbate (e.g., O, CO)")
    args = parser.parse_args()

    logger.info(f"Setting up exploration for {args.adsorbate} on {args.material} ({args.facet})")
    
    # Parse material (simple dummy parser for demo)
    # In reality, use pymatgen Composition
    bulk_dict = {args.material: 1.0} # Placeholder
    
    # Parse facet
    try:
        facet_tuple = tuple(int(x) for x in args.facet)
    except:
        facet_tuple = (0, 0, 1)

    config = {
        "name": f"Explore_{args.material}_{args.facet}_{args.adsorbate}",
        "description": f"Domain-specific exploration of {args.adsorbate} on {args.material}({args.facet})",
        "objective": {
            "type": "adsorption_tuning",
            "adsorbate": args.adsorbate,
            "target_e_ads": -1.5
        },
        "constraints": {
            "bulk": bulk_dict,
            "facet": list(facet_tuple)
        },
        "budget": {"max_evaluations": 10},
        "compute": {"mode": "local_emt"}
    }
    
    run_adsorption_campaign(config)

if __name__ == "__main__":
    main()
