from .state import SurfaceState
from .action import MutationAction, ActionType
import copy
from typing import Dict, Any

class TransitionEngine:
    """
    Deterministic transition function: T(S_t, A_t) -> S_{t+1}.
    
    This engine handles state transitions purely in the *configuration space* (the mathematical descriptor).
    It takes a current `SurfaceState` and a `MutationAction`, and returns a new, independent `SurfaceState`.
    
    Crucially, this does NOT generate the 3D atomic structure (that is the job of the `StructureBuilder`). 
    Separating the mathematical state transition from the physical structure generation allows the system 
    to rapidly search and hash millions of configurations without the overhead of ASE/Pymatgen.
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
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")
            
        return new_state

    def _change_termination(self, state: SurfaceState, params: Dict[str, Any]):
        """Updates the surface termination descriptor (e.g., 'TiO2' to 'SrO')."""
        state.termination = params["termination"]

    def _introduce_vacancy(self, state: SurfaceState, params: Dict[str, Any]):
        """Appends a vacancy defect to the state's defect vector (e.g., params={"site": "O"})."""
        state.defects.append({"type": "vacancy", **params})

    def _substitutional_dopant(self, state: SurfaceState, params: Dict[str, Any]):
        """Appends a substitution defect to the state's defect vector (e.g., params={"original_element": "La", "dopant": "Sr"})."""
        state.defects.append({"type": "substitution", **params})

    def _change_adsorbate(self, state: SurfaceState, params: Dict[str, Any]):
        """Updates the target adsorbate identity (e.g., params={"adsorbate": "OH"})."""
        state.adsorbate = params["adsorbate"]

    def _modify_coverage(self, state: SurfaceState, params: Dict[str, Any]):
        """Updates the surface coverage fraction (e.g., params={"coverage": 0.5})."""
        state.coverage = params["coverage"]

    def _alter_charge_state(self, state: SurfaceState, params: Dict[str, Any]):
        """
        Updates the external electrochemical potential (Phi), used in 
        Computational Hydrogen Electrode (CHE) calculations.
        """
        state.external_conditions["Phi"] = params.get("Phi", state.external_conditions.get("Phi", 0.0))
