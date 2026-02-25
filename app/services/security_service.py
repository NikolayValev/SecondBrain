"""
Security service: startup self-checks and runtime security report.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.config import config
from app.api.models.security import SecurityCheckResult, SecurityReportResponse


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class SecurityService:
    """Evaluates deployment security settings."""

    @staticmethod
    def _is_local_host(host: str) -> bool:
        normalized = host.strip().lower()
        return normalized in {"127.0.0.1", "localhost"}

    def get_report(self) -> SecurityReportResponse:
        public_mode = bool(config.PUBLIC_API_MODE)
        checks: list[SecurityCheckResult] = []

        def add(name: str, status: str, message: str, current: str | None = None, recommendation: str | None = None):
            checks.append(
                SecurityCheckResult(
                    name=name,
                    status=status,
                    message=message,
                    current_value=current,
                    recommendation=recommendation,
                )
            )

        # API key enforcement
        if public_mode and not config.REQUIRE_API_KEY:
            add(
                "require_api_key",
                "fail",
                "Public mode requires API key enforcement.",
                current=str(config.REQUIRE_API_KEY),
                recommendation="Set REQUIRE_API_KEY=true",
            )
        elif not config.REQUIRE_API_KEY:
            add(
                "require_api_key",
                "warn",
                "API key enforcement is disabled.",
                current=str(config.REQUIRE_API_KEY),
                recommendation="Set REQUIRE_API_KEY=true for non-local deployments",
            )
        else:
            add("require_api_key", "pass", "API key enforcement is enabled.", current=str(config.REQUIRE_API_KEY))

        # API key quality
        key_len = len(config.BRAIN_API_KEY or "")
        if config.REQUIRE_API_KEY and key_len < 24:
            add(
                "api_key_strength",
                "fail",
                "API key is too short for secure deployment.",
                current=f"length={key_len}",
                recommendation="Use a random API key of at least 24 characters",
            )
        elif config.REQUIRE_API_KEY:
            add("api_key_strength", "pass", "API key length is acceptable.", current=f"length={key_len}")
        else:
            add("api_key_strength", "warn", "API key not required in current mode.", current=f"length={key_len}")

        # Public exposure switches
        if public_mode and config.EXPOSE_API_DOCS:
            add(
                "expose_api_docs",
                "fail",
                "API docs are public in public mode.",
                current=str(config.EXPOSE_API_DOCS),
                recommendation="Set EXPOSE_API_DOCS=false",
            )
        else:
            status = "warn" if config.EXPOSE_API_DOCS else "pass"
            msg = "API docs are exposed." if config.EXPOSE_API_DOCS else "API docs are not publicly exposed."
            add("expose_api_docs", status, msg, current=str(config.EXPOSE_API_DOCS))

        if public_mode and config.EXPOSE_CONFIG_PUBLIC:
            add(
                "expose_config_public",
                "fail",
                "Config endpoint is public in public mode.",
                current=str(config.EXPOSE_CONFIG_PUBLIC),
                recommendation="Set EXPOSE_CONFIG_PUBLIC=false",
            )
        else:
            status = "warn" if config.EXPOSE_CONFIG_PUBLIC else "pass"
            msg = "Config endpoint is public." if config.EXPOSE_CONFIG_PUBLIC else "Config endpoint is protected."
            add("expose_config_public", status, msg, current=str(config.EXPOSE_CONFIG_PUBLIC))

        # Debug
        if public_mode and config.DEBUG:
            add(
                "debug_mode",
                "fail",
                "Debug mode is enabled in public mode.",
                current=str(config.DEBUG),
                recommendation="Set DEBUG=false",
            )
        else:
            add("debug_mode", "pass", "Debug mode is disabled." if not config.DEBUG else "Debug mode enabled for local use.", current=str(config.DEBUG))

        # Host header hardening
        hosts = [h.strip().lower() for h in config.ALLOWED_HOSTS if h.strip()]
        if "*" in hosts:
            add(
                "allowed_hosts",
                "fail",
                "Wildcard host allowance is unsafe.",
                current="*",
                recommendation="Use explicit hostnames in ALLOWED_HOSTS",
            )
        elif public_mode and all(self._is_local_host(h) for h in hosts):
            add(
                "allowed_hosts",
                "fail",
                "Public mode requires at least one non-local trusted host.",
                current=",".join(hosts),
                recommendation="Include your public hostname in ALLOWED_HOSTS",
            )
        else:
            add("allowed_hosts", "pass", "Trusted hosts are explicitly configured.", current=",".join(hosts))

        # CORS hardening
        origins = [o.strip() for o in config.ALLOWED_ORIGINS if o.strip()]
        if "*" in origins:
            add(
                "allowed_origins",
                "fail",
                "Wildcard CORS origin is unsafe.",
                current="*",
                recommendation="Use explicit trusted origins in ALLOWED_ORIGINS",
            )
        else:
            add("allowed_origins", "pass", "CORS origins are explicit.", current=",".join(origins))

        if public_mode and config.CORS_ALLOW_CREDENTIALS:
            add(
                "cors_allow_credentials",
                "fail",
                "Credentialed CORS is enabled in public mode.",
                current=str(config.CORS_ALLOW_CREDENTIALS),
                recommendation="Set CORS_ALLOW_CREDENTIALS=false unless strictly required",
            )
        else:
            status = "warn" if config.CORS_ALLOW_CREDENTIALS else "pass"
            msg = "Credentialed CORS enabled." if config.CORS_ALLOW_CREDENTIALS else "Credentialed CORS disabled."
            add("cors_allow_credentials", status, msg, current=str(config.CORS_ALLOW_CREDENTIALS))

        # Rate limiting
        if public_mode and not config.RATE_LIMIT_ENABLED:
            add(
                "rate_limit_enabled",
                "fail",
                "Rate limiting is disabled in public mode.",
                current=str(config.RATE_LIMIT_ENABLED),
                recommendation="Set RATE_LIMIT_ENABLED=true",
            )
        elif not config.RATE_LIMIT_ENABLED:
            add(
                "rate_limit_enabled",
                "warn",
                "Rate limiting is disabled.",
                current=str(config.RATE_LIMIT_ENABLED),
                recommendation="Enable rate limiting for internet-facing deployments",
            )
        else:
            add(
                "rate_limit_enabled",
                "pass",
                "Rate limiting is enabled.",
                current=str(config.RATE_LIMIT_ENABLED),
            )

        # Request payload guardrail
        max_request_bytes = int(config.MAX_REQUEST_BYTES)
        if public_mode and max_request_bytes > 2 * 1024 * 1024:
            add(
                "max_request_bytes",
                "fail",
                "Request body limit is too high for public mode.",
                current=str(max_request_bytes),
                recommendation="Set MAX_REQUEST_BYTES to 1048576 (1 MiB) or lower",
            )
        elif max_request_bytes > 1024 * 1024:
            add(
                "max_request_bytes",
                "warn",
                "Request body limit exceeds the recommended default.",
                current=str(max_request_bytes),
                recommendation="Set MAX_REQUEST_BYTES to 1048576 unless larger payloads are required",
            )
        else:
            add(
                "max_request_bytes",
                "pass",
                "Request body limit is within recommended bounds.",
                current=str(max_request_bytes),
            )

        if public_mode and config.API_HOST not in {"127.0.0.1", "localhost"}:
            add(
                "api_bind_host",
                "fail",
                "Public mode should bind API locally and publish via a reverse proxy/tunnel.",
                current=config.API_HOST,
                recommendation="Set API_HOST=127.0.0.1 when using public tunnel/edge",
            )
        else:
            add("api_bind_host", "pass", "API bind host is acceptable.", current=config.API_HOST)

        failed = sum(1 for c in checks if c.status == "fail")
        warned = sum(1 for c in checks if c.status == "warn")
        safe = failed == 0

        return SecurityReportResponse(
            mode="public" if public_mode else "local",
            fail_fast=public_mode,
            safe=safe,
            checked_at=_now_utc_iso(),
            failed_checks=failed,
            warning_checks=warned,
            checks=checks,
        )

    def assert_startup_security(self) -> SecurityReportResponse:
        """
        Validate security posture at startup.
        Raises ValueError when fail-fast mode is active and checks fail.
        """
        report = self.get_report()
        if report.fail_fast and not report.safe:
            failed = [f"{c.name}: {c.message}" for c in report.checks if c.status == "fail"]
            summary = "; ".join(failed[:4])
            raise ValueError(f"Security self-check failed: {summary}")
        return report


# Singleton
security_service = SecurityService()
