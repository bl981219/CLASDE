import os
import time
import subprocess
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutonomousWatchdog")

def check_slurm_jobs():
    """Check if any CLASDE related Slurm jobs are running."""
    try:
        res = subprocess.run(["squeue", "-u", os.environ.get("USER", "bliu"), "-h", "-o", "%j"], 
                             capture_output=True, text=True)
        jobs = res.stdout.strip().split('\n')
        return [j for j in jobs if j.startswith("clasde_")]
    except:
        return []

def check_campaign_processes():
    """Check if the loop_cli.py processes are still alive."""
    try:
        res = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
        lines = res.stdout.strip().split('\n')
        return [l for l in lines if "loop_cli.py" in l and "grep" not in l]
    except:
        return []

def monitor_and_restart(configs):
    """Main loop to keep campaigns moving."""
    logger.info("Starting Autonomous Watchdog...")
    
    while True:
        active_processes = check_campaign_processes()
        active_slurm = check_slurm_jobs()
        
        logger.info(f"Status: {len(active_processes)} active loops, {len(active_slurm)} active Slurm jobs.")
        
        # If a process is dead but we still have budget or unfinished business, restart it
        # Note: Since our submit_job has re-attachment logic, restarting is safe.
        for config_path in configs:
            process_running = any(config_path in p for p in active_processes)
            
            if not process_running:
                logger.warning(f"Campaign {config_path} seems to have stopped. Attempting restart...")
                log_name = f"campaign_log_{os.path.basename(config_path).replace('.yaml', '')}.txt"
                cmd = f"PYTHONPATH=. nohup python3 -u cli/loop_cli.py --config {config_path} >> {log_name} 2>&1 &"
                subprocess.Popen(cmd, shell=True)
                logger.info(f"Restarted {config_path}. Log: {log_name}")
        
        time.sleep(600) # Check every 10 minutes

def main():
    parser = argparse.ArgumentParser(description="CLASDE Autonomous Campaign Watchdog")
    parser.add_argument("--configs", nargs="+", help="List of campaign YAML files to keep alive.")
    args = parser.parse_args()
    
    if not args.configs:
        # Default to the three test prompts if none provided
        args.configs = [
            "configs/test_lsf_segregation.yaml",
            "configs/test_lscf_poisoning.yaml",
            "configs/test_sto_doping.yaml"
        ]
        
    monitor_and_restart(args.configs)

if __name__ == "__main__":
    main()
