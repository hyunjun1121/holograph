"""
HOLOGRAPH Custom Exceptions.

Defines custom error types for operational safety and control.
"""


class HolographError(Exception):
    """Base exception for HOLOGRAPH."""
    pass


class BudgetExceededError(HolographError):
    """
    Raised when a resource budget (tokens, queries) is exceeded.

    This is a P0 (Priority 0) operational safety mechanism to prevent
    unbounded API costs during experiments.
    """

    def __init__(self, message: str, budget_type: str = "unknown", current: int = 0, limit: int = 0):
        """
        Initialize BudgetExceededError.

        Args:
            message: Human-readable error message
            budget_type: Type of budget exceeded ("tokens" or "queries")
            current: Current usage count
            limit: Budget limit that was exceeded
        """
        self.budget_type = budget_type
        self.current = current
        self.limit = limit
        super().__init__(message)

    def __str__(self):
        return f"BudgetExceededError({self.budget_type}): {self.args[0]} (current={self.current}, limit={self.limit})"


class ConfigurationError(HolographError):
    """Raised when there is a configuration issue."""
    pass


class LLMError(HolographError):
    """Raised when there is an LLM API error."""
    pass


class SheafConstraintError(HolographError):
    """Raised when sheaf constraints cannot be satisfied."""
    pass
