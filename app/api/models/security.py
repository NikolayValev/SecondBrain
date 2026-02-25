"""
Security-related API models.
"""

from typing import Optional, Literal

from pydantic import BaseModel


class SecurityCheckResult(BaseModel):
    """Single security check outcome."""
    name: str
    status: Literal["pass", "warn", "fail"]
    message: str
    current_value: Optional[str] = None
    recommendation: Optional[str] = None


class SecurityReportResponse(BaseModel):
    """Aggregated security self-check report."""
    mode: str
    fail_fast: bool
    safe: bool
    checked_at: str
    failed_checks: int
    warning_checks: int
    checks: list[SecurityCheckResult]
