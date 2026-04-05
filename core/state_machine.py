"""
State machine for campaign lifecycle management.

Provides explicit state transitions with validation to prevent invalid states.
"""

import logging
from enum import Enum
from typing import Dict, List, Callable, Optional
from core.exceptions import StateTransitionError

logger = logging.getLogger(__name__)


class CampaignState(str, Enum):
    """Campaign lifecycle states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"


class CampaignStateMachine:
    """
    Explicit state machine for campaign lifecycle.
    
    Enforces valid state transitions and provides hooks for
    state entry/exit callbacks.
    """
    
    # Valid transitions from each state
    TRANSITIONS: Dict[CampaignState, List[CampaignState]] = {
        CampaignState.INITIALIZING: [
            CampaignState.RUNNING,
            CampaignState.FAILED
        ],
        CampaignState.RUNNING: [
            CampaignState.PAUSED,
            CampaignState.COMPLETING,
            CampaignState.FAILED
        ],
        CampaignState.PAUSED: [
            CampaignState.RUNNING,
            CampaignState.COMPLETING,
            CampaignState.FAILED
        ],
        CampaignState.COMPLETING: [
            CampaignState.COMPLETED,
            CampaignState.FAILED
        ],
        CampaignState.COMPLETED: [],  # Terminal state
        CampaignState.FAILED: []  # Terminal state
    }
    
    def __init__(self, initial_state: CampaignState = CampaignState.INITIALIZING):
        self.state = initial_state
        self._on_enter: Dict[CampaignState, List[Callable]] = {}
        self._on_exit: Dict[CampaignState, List[Callable]] = {}
        self._transition_history: List[tuple] = []
    
    def transition_to(self, new_state: CampaignState) -> None:
        """
        Transition to a new state with validation.
        
        Args:
            new_state: Target state
            
        Raises:
            StateTransitionError: If transition is invalid
        """
        if not self.can_transition_to(new_state):
            raise StateTransitionError(
                f"Invalid transition: {self.state} -> {new_state}",
                details={
                    "current_state": self.state,
                    "attempted_state": new_state,
                    "valid_transitions": self.TRANSITIONS[self.state]
                }
            )
        
        old_state = self.state
        
        # Execute exit callbacks
        self._execute_callbacks(self._on_exit.get(old_state, []))
        
        # Change state
        self.state = new_state
        
        # Record transition
        import time
        self._transition_history.append((old_state, new_state, time.time()))
        
        # Execute entry callbacks
        self._execute_callbacks(self._on_enter.get(new_state, []))
        
        logger.info(f"State transition: {old_state} → {new_state}")
    
    def can_transition_to(self, state: CampaignState) -> bool:
        """
        Check if transition to state is valid.
        
        Args:
            state: Target state
            
        Returns:
            True if transition is valid
        """
        return state in self.TRANSITIONS[self.state]
    
    def on_enter(self, state: CampaignState, callback: Callable) -> None:
        """
        Register callback for state entry.
        
        Args:
            state: State to register callback for
            callback: Function to call on entry
        """
        if state not in self._on_enter:
            self._on_enter[state] = []
        self._on_enter[state].append(callback)
    
    def on_exit(self, state: CampaignState, callback: Callable) -> None:
        """
        Register callback for state exit.
        
        Args:
            state: State to register callback for
            callback: Function to call on exit
        """
        if state not in self._on_exit:
            self._on_exit[state] = []
        self._on_exit[state].append(callback)
    
    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        return len(self.TRANSITIONS[self.state]) == 0
    
    def is_running(self) -> bool:
        """Check if campaign is actively running."""
        return self.state == CampaignState.RUNNING
    
    def is_paused(self) -> bool:
        """Check if campaign is paused."""
        return self.state == CampaignState.PAUSED
    
    def is_completed(self) -> bool:
        """Check if campaign completed successfully."""
        return self.state == CampaignState.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if campaign failed."""
        return self.state == CampaignState.FAILED
    
    def get_history(self) -> List[tuple]:
        """
        Get transition history.
        
        Returns:
            List of (old_state, new_state, timestamp) tuples
        """
        return self._transition_history.copy()
    
    def _execute_callbacks(self, callbacks: List[Callable]) -> None:
        """Execute callbacks with error handling."""
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(
                    f"State callback failed: {callback.__name__}",
                    exc_info=True
                )
                # Don't let callback failures break state transitions
