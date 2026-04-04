import logging
import json
import time
from typing import Any, Dict

class StructuredLogger:
    def __init__(self, name: str, log_file: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Avoid adding multiple handlers if already initialized
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            self.logger.addHandler(handler)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data
        }
        self.logger.info(json.dumps(event))

# Global observability instance
obs_logger = StructuredLogger("clasde_observability", "data/results/telemetry.jsonl")

def log_agent_decision(agent_name: str, input_data: Dict[str, Any], output_data: Dict[str, Any], reasoning: str = ""):
    obs_logger.log_event("agent_decision", {
        "agent": agent_name,
        "input": input_data,
        "output": output_data,
        "reasoning": reasoning
    })
