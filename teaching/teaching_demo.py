import os
import shutil
import sys
from datetime import datetime

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from execution.workflow_runner import run_adsorption_campaign

def run_teaching_demo():
    """
    Runs a minimal 2-iteration campaign and preserves logs for teaching.
    """
    print("\n" + "="*60)
    print("   CLASDE TEACHING DEMO: SYSTEM WALKTHROUGH")
    print("="*60)
    
    # 1. Define a minimal teaching configuration
    teaching_config = {
        "name": "Teaching_Demo_Cu111",
        "description": "Minimal 2-iteration test for teaching user about discovery logs.",
        "original_prompt": "teach me about oxygen on copper 111",
        "objective": {
            "type": "adsorption_tuning",
            "adsorbate": "O",
            "target_e_ads": -1.5
        },
        "constraints": {
            "bulk": {"Cu": 1.0},
            "facet": [1, 1, 1]
        },
        "budget": {"max_evaluations": 2},
        "acquisition": {"acquisition_type": "EI", "kappa": 2.576},
        "compute": {"mode": "local_emt"}
    }

    # 2. Cleanup previous data for a fresh demo
    results_dir = "data/results"
    output_dir = "teaching/teaching_results"
    
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Execute the autonomous loop
    print("\n[Phase 1] Executing 2-iteration Autonomous Campaign...")
    run_adsorption_campaign(teaching_config)

    # 4. Preserve logs for teaching
    print("\n[Phase 2] Preserving logs for teaching...")
    log_file = os.path.join(results_dir, "research_log.md")
    memory_file = os.path.join(results_dir, "clasde_memory.json")
    
    if os.path.exists(log_file):
        shutil.copy(log_file, os.path.join(output_dir, "teaching_research_log.md"))
        print(f"  - Copied {log_file} to {output_dir}")
        
    if os.path.exists(memory_file):
        shutil.copy(memory_file, os.path.join(output_dir, "teaching_memory.json"))
        print(f"  - Copied {memory_file} to {output_dir}")

    # 5. Create the Teaching Guide
    teaching_guide_path = os.path.join(output_dir, "TEACHING_GUIDE.md")
    with open(teaching_guide_path, "w") as f:
        f.write("# CLASDE Teaching Guide: Understanding the Research Log\n\n")
        f.write("This directory contains the output of a minimal 2-iteration research campaign. ")
        f.write("Use these files to teach users how CLASDE documents its discovery process.\n\n")
        
        f.write("## 1. The Research Log (`teaching_research_log.md`)\n")
        f.write("This file is the human-readable 'Master Record' of the campaign. Key sections include:\n")
        f.write("- **Original User Intent:** Shows the natural language prompt that started the research.\n")
        f.write("- **Exploration Phase:** A tabular record of every experiment, the action taken, and the physical reward observed.\n")
        f.write("- **Scientific Reasoning Phase:** Contains theories discovered by the PI (Principal Investigator) Agent.\n\n")
        
        f.write("## 2. The Memory Graph (`teaching_memory.json`)\n")
        f.write("This is the machine-readable 'Digital Lab Notebook'. It stores:\n")
        f.write("- Canonical descriptors of every surface configuration visited.\n")
        f.write("- The exact atomic structures used in calculations.\n")
        f.write("- Statistical patterns used for training future surrogate models.\n\n")
        
        f.write("## 3. How to use this for training\n")
        f.write("1. Open `teaching_research_log.md` in a Markdown viewer.\n")
        f.write("2. Trace how the system interpreted the prompt into a mathematical objective.\n")
        f.write("3. Observe how 'Vacancy_Density' or 'Coverage' were identified as driving factors in the theory section.\n")

    print(f"\n[Phase 3] Created Teaching Guide at {teaching_guide_path}")
    print("\nTeaching Demo Complete. Results preserved in teaching/teaching_results/\n")

if __name__ == "__main__":
    run_teaching_demo()
