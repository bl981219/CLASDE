"""
Custom exception hierarchy for CLASDE.

Provides structured error handling throughout the system.
"""


class CLASDEException(Exception):
    """Base exception for all CLASDE-specific errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class LLMAPIError(CLASDEException):
    """Raised when LLM API calls fail."""
    pass


class ComputationError(CLASDEException):
    """Raised when DFT/MLIP calculations fail."""
    pass


class ValidationError(CLASDEException):
    """Raised when physical or chemical validation fails."""
    pass


class MemoryError(CLASDEException):
    """Raised when database or storage operations fail."""
    pass


class ConfigurationError(CLASDEException):
    """Raised when configuration is invalid."""
    pass


class StateTransitionError(CLASDEException):
    """Raised when invalid state transitions are attempted."""
    pass


class AgentError(CLASDEException):
    """Raised when agent operations fail."""
    pass


class BackendError(CLASDEException):
    """Raised when execution backend operations fail."""
    pass
