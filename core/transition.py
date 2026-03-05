from .state import SurfaceState
from .action import MutationAction, ActionType
import copy
from typing import Dict, Any

class TransitionEngine:
    """
    Deterministic transition function T(S_t, A_t) -> S_{t+1}.
    This engine handles the state transitions in the configuration space.
    The actual structural generation (ASE/Pymatgen) should be coordinated
    via the StructureBuilder agent using this state.
    """
    def apply(self, state: SurfaceState, action: MutationAction) -> SurfaceState:
        """Apply a mutation action to a state to produce a new state."""
        # Create a deep copy of the state
        new_state = state.model_copy(deep=True)
        
        # Dispatch to specific mutation logic
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
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")
            
        return new_state

    def _change_termination(self, state: SurfaceState, params: Dict[str, Any]):
        state.termination = params["termination"]

    def _introduce_vacancy(self, state: SurfaceState, params: Dict[str, Any]):
        # e.g., {"site": "O1", "index": 0}
        state.defects.append({"type": "vacancy", **params})

    def _substitutional_dopant(self, state: SurfaceState, params: Dict[str, Any]):
        # e.g., {"original_element": "O", "dopant": "N", "index": 0}
        state.defects.append({"type": "substitution", **params})

    def _change_adsorbate(self, state: SurfaceState, params: Dict[str, Any]):
        state.adsorbate = params["adsorbate"]

    def _modify_coverage(self, state: SurfaceState, params: Dict[str, Any]):
        state.coverage = params["coverage"]

    def _alter_charge_state(self, state: SurfaceState, params: Dict[str, Any]):
        # Phi represents the electrode potential or charge state
        state.external_conditions["Phi"] = params.get("Phi", state.external_conditions.get("Phi", 0.0))
