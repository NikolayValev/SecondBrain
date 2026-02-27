"""
Tests for startup security self-check service.
"""

from app.config import config
from app.services.security_service import security_service


def _set_secure_public_defaults(monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_API_MODE", True)
    monkeypatch.setattr(config, "REQUIRE_API_KEY", True)
    monkeypatch.setattr(config, "BRAIN_API_KEY", "x" * 32)
    monkeypatch.setattr(config, "EXPOSE_API_DOCS", False)
    monkeypatch.setattr(config, "EXPOSE_CONFIG_PUBLIC", False)
    monkeypatch.setattr(config, "DEBUG", False)
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ["brain.nikolayvalev.com", "127.0.0.1"])
    monkeypatch.setattr(config, "ALLOWED_ORIGINS", ["https://brain.nikolayvalev.com"])
    monkeypatch.setattr(config, "CORS_ALLOW_CREDENTIALS", False)
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(config, "MAX_REQUEST_BYTES", 1024 * 1024)
    monkeypatch.setattr(config, "API_HOST", "127.0.0.1")


def _check(report, name: str):
    for check in report.checks:
        if check.name == name:
            return check
    raise AssertionError(f"Missing check: {name}")


def test_security_report_safe_in_public_mode(monkeypatch):
    _set_secure_public_defaults(monkeypatch)
    report = security_service.get_report()
    assert report.mode == "public"
    assert report.fail_fast is True
    assert report.safe is True
    assert report.failed_checks == 0


def test_security_report_fails_when_docs_exposed_in_public_mode(monkeypatch):
    _set_secure_public_defaults(monkeypatch)
    monkeypatch.setattr(config, "EXPOSE_API_DOCS", True)
    report = security_service.get_report()
    assert report.safe is False
    assert report.failed_checks >= 1
    assert any(c.name == "expose_api_docs" and c.status == "fail" for c in report.checks)


def test_assert_startup_security_raises_in_public_mode(monkeypatch):
    _set_secure_public_defaults(monkeypatch)
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ["*"])
    try:
        security_service.assert_startup_security()
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Security self-check failed" in str(exc)


def test_assert_startup_security_does_not_raise_in_local_mode(monkeypatch):
    _set_secure_public_defaults(monkeypatch)
    monkeypatch.setattr(config, "PUBLIC_API_MODE", False)
    monkeypatch.setattr(config, "REQUIRE_API_KEY", False)
    monkeypatch.setattr(config, "BRAIN_API_KEY", "")
    report = security_service.assert_startup_security()
    assert report.mode == "local"


def test_security_report_fails_when_rate_limit_disabled_in_public_mode(monkeypatch):
    _set_secure_public_defaults(monkeypatch)
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", False)
    report = security_service.get_report()
    assert report.safe is False
    assert any(c.name == "rate_limit_enabled" and c.status == "fail" for c in report.checks)


def test_security_report_fails_when_request_limit_too_high_in_public_mode(monkeypatch):
    _set_secure_public_defaults(monkeypatch)
    monkeypatch.setattr(config, "MAX_REQUEST_BYTES", 8 * 1024 * 1024)
    report = security_service.get_report()
    assert report.safe is False
    assert any(c.name == "max_request_bytes" and c.status == "fail" for c in report.checks)


def test_security_report_fails_when_config_exposed_in_public_mode(monkeypatch):
    _set_secure_public_defaults(monkeypatch)
    monkeypatch.setattr(config, "EXPOSE_CONFIG_PUBLIC", True)
    report = security_service.get_report()
    check = _check(report, "expose_config_public")
    assert report.safe is False
    assert check.status == "fail"


def test_security_report_fails_for_wildcard_allowed_hosts(monkeypatch):
    _set_secure_public_defaults(monkeypatch)
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ["*"])
    report = security_service.get_report()
    check = _check(report, "allowed_hosts")
    assert report.safe is False
    assert check.status == "fail"


def test_security_report_fails_when_public_mode_has_only_local_hosts(monkeypatch):
    _set_secure_public_defaults(monkeypatch)
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ["127.0.0.1", "localhost"])
    report = security_service.get_report()
    check = _check(report, "allowed_hosts")
    assert report.safe is False
    assert check.status == "fail"


def test_security_report_fails_for_wildcard_allowed_origins(monkeypatch):
    _set_secure_public_defaults(monkeypatch)
    monkeypatch.setattr(config, "ALLOWED_ORIGINS", ["*"])
    report = security_service.get_report()
    check = _check(report, "allowed_origins")
    assert report.safe is False
    assert check.status == "fail"


def test_security_report_fails_for_credentialed_cors_in_public_mode(monkeypatch):
    _set_secure_public_defaults(monkeypatch)
    monkeypatch.setattr(config, "CORS_ALLOW_CREDENTIALS", True)
    report = security_service.get_report()
    check = _check(report, "cors_allow_credentials")
    assert report.safe is False
    assert check.status == "fail"


def test_security_report_fails_when_public_mode_binds_non_local_host(monkeypatch):
    _set_secure_public_defaults(monkeypatch)
    monkeypatch.setattr(config, "API_HOST", "0.0.0.0")
    report = security_service.get_report()
    check = _check(report, "api_bind_host")
    assert report.safe is False
    assert check.status == "fail"


def test_security_report_warns_when_docs_exposed_in_local_mode(monkeypatch):
    _set_secure_public_defaults(monkeypatch)
    monkeypatch.setattr(config, "PUBLIC_API_MODE", False)
    monkeypatch.setattr(config, "EXPOSE_API_DOCS", True)
    report = security_service.get_report()
    check = _check(report, "expose_api_docs")
    assert check.status == "warn"
    assert report.warning_checks >= 1
