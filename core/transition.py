import logging
import copy
from typing import Dict, Any
from .state import SurfaceState, AdsorbateInstance
from .action import MutationAction, ActionType

logger = logging.getLogger(__name__)

class TransitionEngine:
    """
    Deterministic transition function: T(S_t, A_t) -> S_{t+1}.
    
    This engine handles state transitions purely in the *configuration space* (the mathematical descriptor).
    It takes a current `SurfaceState` and a `MutationAction`, and returns a new, independent `SurfaceState`.
    """
    def apply(self, state: SurfaceState, action: MutationAction) -> SurfaceState:
        """
        Apply a mutation action to a state to produce a new state.
        A deep copy is created to ensure immutability of the historical graph.
        """
        # Create a deep copy of the state to preserve history
        new_state = state.model_copy(deep=True)
        
        # Dispatch to specific mutation logic based on the action type
        if action.action_type == ActionType.CHANGE_TERMINATION:
            self._change_termination(new_state, action.parameters)
        elif action.action_type == ActionType.INTRODUCE_VACANCY:
            self._introduce_vacancy(new_state, action.parameters)
        elif action.action_type == ActionType.SUBSTITUTIONAL_DOPANT:
            self._substitutional_dopant(new_state, action.parameters)
        elif action.action_type == ActionType.CHANGE_ADSORBATE:
            self._change_adsorbate(new_state, action.parameters)
        elif action.action_type == ActionType.MODIFY_COVERAGE:
            self._modify_coverage(new_state, action.parameters)
        elif action.action_type == ActionType.ALTER_CHARGE_STATE:
            self._alter_charge_state(new_state, action.parameters)
        elif action.action_type == ActionType.SWAP_ATOMS:
            self._swap_atoms(new_state, action.parameters)
        else:
            logger.error(f"Unknown action type: {action.action_type}")
            raise ValueError(f"Unknown action type: {action.action_type}")
            
        return new_state

    def _change_termination(self, state: SurfaceState, params: Dict[str, Any]) -> None:
        """Updates the surface termination descriptor (e.g., 'TiO2' to 'SrO')."""
        state.termination = params["termination"]

    def _introduce_vacancy(self, state: SurfaceState, params: Dict[str, Any]) -> None:
        """Appends a vacancy defect to the state's defect vector (e.g., params={"site": "O"})."""
        state.defects.append({"type": "vacancy", **params})

    def _substitutional_dopant(self, state: SurfaceState, params: Dict[str, Any]) -> None:
        """Appends a substitution defect to the state's defect vector (e.g., params={"original_element": "La", "dopant": "Sr"})."""
        state.defects.append({"type": "substitution", **params})

    def _change_adsorbate(self, state: SurfaceState, params: Dict[str, Any]) -> None:
        """Updates the target adsorbate identity (e.g., params={"adsorbate": "OH"})."""
        # For simplicity, we replace all adsorbates or just the first one
        if state.adsorbates:
            state.adsorbates[0].identity = params["adsorbate"]
        else:
            state.adsorbates.append(AdsorbateInstance(identity=params["adsorbate"], coverage=0.25))
        
        # Keep summary field updated
        state.coverage = sum(a.coverage for a in state.adsorbates)

    def _modify_coverage(self, state: SurfaceState, params: Dict[str, Any]) -> None:
        """Updates the surface coverage fraction (e.g., params={"coverage": 0.5})."""
        if state.adsorbates:
            state.adsorbates[0].coverage = params["coverage"]
        else:
            # Assume some default if list is empty
            state.adsorbates.append(AdsorbateInstance(identity="O", coverage=params["coverage"]))
            
        state.coverage = sum(a.coverage for a in state.adsorbates)

    def _alter_charge_state(self, state: SurfaceState, params: Dict[str, Any]) -> None:
        """
        Updates the external electrochemical potential (Phi), used in 
        Computational Hydrogen Electrode (CHE) calculations.
        """
        state.external_conditions["Phi"] = params.get("Phi", state.external_conditions.get("Phi", 0.0))

    def _swap_atoms(self, state: SurfaceState, params: Dict[str, Any]) -> None:
        """Records an atomic swap as a complex defect (e.g., params={"element_a": "La", "element_b": "Sr"})."""
        state.defects.append({"type": "swap", **params})
