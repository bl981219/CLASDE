import time
import json
import logging
import os
from typing import Dict, Any, Optional

class TelemetrySystem:
    """
    Standardized observability for the CLASDE discovery engine.
    Logs system performance, agent decisions, and execution metrics.
    """
    def __init__(self, log_dir: str = "data/results"):
        self.log_path = os.path.join(log_dir, "telemetry.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure file logger
        self.logger = logging.getLogger("clasde_telemetry")
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_path)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log_metric(self, metric_name: str, value: Any, metadata: Optional[Dict] = None):
        """Record a numerical or categorical metric."""
        entry = {
            "timestamp": time.time(),
            "type": "metric",
            "name": metric_name,
            "value": value,
            "metadata": metadata or {}
        }
        self.logger.info(json.dumps(entry))

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Record a system event (e.g. 'job_submitted', 'hypothesis_falsified')."""
        entry = {
            "timestamp": time.time(),
            "type": "event",
            "event": event_type,
            "data": data
        }
        self.logger.info(json.dumps(entry))

# Global Telemetry Instance
telemetry = TelemetrySystem()
