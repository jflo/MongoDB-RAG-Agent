"""Tests for error handling utilities."""

import pytest
from httpx import ConnectError, ConnectTimeout, ReadTimeout, HTTPStatusError, Response, Request
from src.errors import (
    classify_llm_error,
    format_error_for_cli,
    format_error_for_slack,
    is_retryable_error
)


class TestClassifyLLMError:
    """Test error classification."""

    def test_connection_refused(self):
        """Connection refused should be classified correctly."""
        error = ConnectionError("Connection refused")
        short, detailed = classify_llm_error(error)
        assert "unreachable" in short.lower() or "connection" in short.lower()
        assert "LLM_BASE_URL" in detailed

    def test_timeout_error(self):
        """Timeout errors should be classified correctly."""
        error = TimeoutError("Connection timed out")
        short, detailed = classify_llm_error(error)
        assert "timeout" in short.lower() or "connection" in short.lower()

    def test_api_key_error(self):
        """API key errors should be classified correctly."""
        error = Exception("Invalid API key provided")
        short, detailed = classify_llm_error(error)
        assert "auth" in short.lower() or "key" in short.lower()

    def test_model_not_found(self):
        """Model not found errors should be classified correctly."""
        error = Exception("Model 'gpt-5' does not exist")
        short, detailed = classify_llm_error(error)
        assert "model" in short.lower()
        assert "LLM_MODEL" in detailed

    def test_dns_resolution_error(self):
        """DNS resolution errors should be classified correctly."""
        error = Exception("nodename nor servname provided")
        short, detailed = classify_llm_error(error)
        assert "host" in short.lower() or "connection" in short.lower()

    def test_unknown_error_fallback(self):
        """Unknown errors should still return a message."""
        error = Exception("Something completely unexpected")
        short, detailed = classify_llm_error(error)
        assert short  # Should have some message
        assert detailed  # Should have some detail


class TestFormatErrorForCLI:
    """Test CLI error formatting."""

    def test_format_includes_markup(self):
        """CLI format should include Rich markup."""
        error = Exception("Test error")
        formatted = format_error_for_cli(error)
        assert "[red" in formatted or "[dim" in formatted

    def test_format_is_string(self):
        """CLI format should return a string."""
        error = Exception("Test error")
        formatted = format_error_for_cli(error)
        assert isinstance(formatted, str)


class TestFormatErrorForSlack:
    """Test Slack error formatting."""

    def test_format_includes_emoji(self):
        """Slack format should include warning emoji."""
        error = Exception("Test error")
        formatted = format_error_for_slack(error)
        assert "⚠️" in formatted

    def test_format_includes_bold(self):
        """Slack format should include bold markers."""
        error = Exception("Test error")
        formatted = format_error_for_slack(error)
        assert "*" in formatted  # Slack bold syntax


class TestIsRetryableError:
    """Test retry classification."""

    def test_timeout_is_retryable(self):
        """Timeout errors should be retryable."""
        error = Exception("Connection timed out")
        assert is_retryable_error(error) is True

    def test_auth_not_retryable(self):
        """Auth errors should not be retryable."""
        error = Exception("Invalid API key")
        assert is_retryable_error(error) is False

    def test_config_not_retryable(self):
        """Configuration errors should not be retryable."""
        error = Exception("Model does not exist")
        assert is_retryable_error(error) is False


class TestHTTPStatusErrors:
    """Test HTTP status error classification."""

    def _make_http_error(self, status_code: int) -> HTTPStatusError:
        """Helper to create HTTPStatusError."""
        request = Request("GET", "https://api.example.com")
        response = Response(status_code, request=request)
        return HTTPStatusError("Error", request=request, response=response)

    def test_401_unauthorized(self):
        """401 should indicate auth failure."""
        error = self._make_http_error(401)
        short, detailed = classify_llm_error(error)
        assert "auth" in short.lower()

    def test_403_forbidden(self):
        """403 should indicate access denied."""
        error = self._make_http_error(403)
        short, detailed = classify_llm_error(error)
        assert "denied" in short.lower() or "access" in short.lower()

    def test_404_not_found(self):
        """404 should indicate endpoint not found."""
        error = self._make_http_error(404)
        short, detailed = classify_llm_error(error)
        assert "not found" in short.lower()

    def test_429_rate_limited(self):
        """429 should indicate rate limiting."""
        error = self._make_http_error(429)
        short, detailed = classify_llm_error(error)
        assert "rate" in short.lower()
        assert is_retryable_error(error) is True

    def test_500_server_error(self):
        """500 should indicate server error."""
        error = self._make_http_error(500)
        short, detailed = classify_llm_error(error)
        assert "error" in short.lower()
        assert is_retryable_error(error) is True

    def test_503_service_unavailable(self):
        """503 should be retryable."""
        error = self._make_http_error(503)
        assert is_retryable_error(error) is True
