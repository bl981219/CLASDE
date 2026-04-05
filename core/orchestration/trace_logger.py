"""
Knowledge trace logger for scientific auditability.

Logs complete provenance of all knowledge transformations
(Idea → Hypothesis → Experiment → Insight).
"""

import logging
import os
import json
from typing import List, Any, Optional
from core.lab_objects import KnowledgeTrace
from core.exceptions import MemoryError

logger = logging.getLogger(__name__)


class TraceLogger:
    """
    Manages JSONL-based knowledge trace logging.
    
    Provides full scientific auditability by recording every
    transformation in the knowledge pipeline.
    """
    
    def __init__(self, trace_log_path: str):
        self.trace_log_path = trace_log_path
        
        # Ensure directory exists
        log_dir = os.path.dirname(trace_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def log_trace(self, trace: KnowledgeTrace) -> None:
        """
        Append a knowledge trace to the log.
        
        Args:
            trace: KnowledgeTrace object to log
            
        Raises:
            MemoryError: If trace write fails
        """
        try:
            with open(self.trace_log_path, "a") as f:
                f.write(trace.model_dump_json() + "\n")
            
            logger.debug(f"Knowledge trace logged for iteration {trace.iteration}")
            
        except Exception as e:
            logger.error(f"Failed to log knowledge trace: {e}")
            raise MemoryError(
                f"Trace logging failed: {e}",
                details={
                    "trace_log_path": self.trace_log_path,
                    "iteration": trace.iteration
                }
            )
    
    def get_traces(self, limit: Optional[int] = None) -> List[KnowledgeTrace]:
        """
        Retrieve all or recent knowledge traces.
        
        Args:
            limit: Maximum number of traces to return (most recent)
            
        Returns:
            List of KnowledgeTrace objects
        """
        if not os.path.exists(self.trace_log_path):
            return []
        
        traces = []
        
        try:
            with open(self.trace_log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        trace_data = json.loads(line)
                        traces.append(KnowledgeTrace(**trace_data))
            
            if limit and len(traces) > limit:
                traces = traces[-limit:]
            
            return traces
            
        except Exception as e:
            logger.error(f"Failed to read traces: {e}")
            raise MemoryError(
                f"Trace retrieval failed: {e}",
                details={"trace_log_path": self.trace_log_path}
            )
    
    def get_trace_by_iteration(self, iteration: int) -> Optional[KnowledgeTrace]:
        """
        Retrieve trace for specific iteration.
        
        Args:
            iteration: Iteration number
            
        Returns:
            KnowledgeTrace object or None if not found
        """
        traces = self.get_traces()
        for trace in traces:
            if trace.iteration == iteration:
                return trace
        return None
    
    def count_traces(self) -> int:
        """Count total number of traces."""
        if not os.path.exists(self.trace_log_path):
            return 0
        
        count = 0
        try:
            with open(self.trace_log_path, "r") as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
        except Exception as e:
            logger.error(f"Failed to count traces: {e}")
            return 0
    
    def clear_traces(self) -> None:
        """Clear all traces (use with caution)."""
        if os.path.exists(self.trace_log_path):
            try:
                os.remove(self.trace_log_path)
                logger.warning(f"All traces cleared from {self.trace_log_path}")
            except Exception as e:
                logger.error(f"Failed to clear traces: {e}")
    
    def export_traces_to_json(self, output_path: str) -> None:
        """
        Export all traces to a single JSON file.
        
        Useful for archival or analysis.
        
        Args:
            output_path: Path to output JSON file
        """
        traces = self.get_traces()
        
        try:
            with open(output_path, "w") as f:
                json.dump(
                    [trace.model_dump() for trace in traces],
                    f,
                    indent=2,
                    default=str
                )
            
            logger.info(f"Exported {len(traces)} traces to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export traces: {e}")
            raise MemoryError(
                f"Trace export failed: {e}",
                details={"output_path": output_path}
            )
