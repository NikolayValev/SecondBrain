"""
Security routes: /security/self-check
"""

from fastapi import APIRouter

from app.api.models.security import SecurityReportResponse
from app.services.security_service import security_service

router = APIRouter(tags=["Security"])


@router.get("/security/self-check", response_model=SecurityReportResponse)
async def security_self_check():
    """
    Return runtime security posture report.

    This endpoint is authenticated by default (unless global auth is disabled).
    """
    return security_service.get_report()
