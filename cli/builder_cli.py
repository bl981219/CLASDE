import argparse
import json
import os
from core.state import SurfaceState
from agents.builder_agent import StructureBuilder

def main():
    parser = argparse.ArgumentParser(description="CLASDE Structure Builder CLI")
    parser.add_argument("state_file", type=str, help="Path to JSON file containing SurfaceState.")
    parser.add_argument("--output", type=str, default="POSCAR", help="Output filename (default: POSCAR).")
    parser.add_argument("--format", type=str, default="vasp", help="Output format (default: vasp).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.state_file):
        print(f"Error: {args.state_file} not found.")
        return
        
    with open(args.state_file, "r") as f:
        state_dict = json.load(f)
        
    # Handle both raw model_dump and nested formats
    if "state" in state_dict:
        state = SurfaceState(**state_dict["state"])
    else:
        state = SurfaceState(**state_dict)
        
    print(f"Building structure for state: {state.get_summary()}")
    builder = StructureBuilder()
    atoms = builder.build_structure(state)
    
    if atoms is not None:
        from ase.io import write
        write(args.output, atoms, format=args.format)
        print(f"Structure saved to {args.output}")
    else:
        print("Error: Structure generation failed (check if ASE/Pymatgen are installed).")

if __name__ == "__main__":
    main()
