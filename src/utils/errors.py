"""Error handling utilities for LLM and API connections.

Provides user-friendly error messages for common connection issues.
"""

import logging
from typing import Optional, Tuple
from httpx import ConnectError, ConnectTimeout, ReadTimeout, HTTPStatusError

logger = logging.getLogger(__name__)


class LLMConnectionError(Exception):
    """Raised when LLM provider is unreachable or returns an error."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(message)


def classify_llm_error(error: Exception) -> Tuple[str, str]:
    """
    Classify an LLM-related error and return user-friendly message.

    Args:
        error: The exception that was raised

    Returns:
        Tuple of (short_message, detailed_message)
    """
    error_type = type(error).__name__
    error_str = str(error).lower()

    # Connection refused / host unreachable
    if isinstance(error, ConnectError) or "connection refused" in error_str:
        return (
            "LLM provider unreachable",
            "Cannot connect to the LLM provider. Check that LLM_BASE_URL is correct "
            "and the service is running."
        )

    # Connection timeout
    if isinstance(error, ConnectTimeout) or "connect timeout" in error_str:
        return (
            "LLM connection timeout",
            "Connection to LLM provider timed out. The service may be slow or unreachable. "
            "Check your network connection and LLM_BASE_URL."
        )

    # Read timeout (connected but response too slow)
    if isinstance(error, ReadTimeout) or "read timeout" in error_str:
        return (
            "LLM response timeout",
            "LLM provider is responding slowly. This may be due to high load or a complex query. "
            "Try again or check the provider status."
        )

    # HTTP status errors
    if isinstance(error, HTTPStatusError):
        status_code = error.response.status_code
        if status_code == 401:
            return (
                "LLM authentication failed",
                "Invalid API key. Check that LLM_API_KEY is correct for your provider."
            )
        elif status_code == 403:
            return (
                "LLM access denied",
                "Access denied by LLM provider. Your API key may lack permissions or the model "
                "may not be available on your plan."
            )
        elif status_code == 404:
            return (
                "LLM endpoint not found",
                "LLM endpoint not found. Check that LLM_BASE_URL and LLM_MODEL are correct."
            )
        elif status_code == 429:
            return (
                "LLM rate limited",
                "Too many requests to LLM provider. Wait a moment and try again."
            )
        elif status_code >= 500:
            return (
                "LLM provider error",
                f"LLM provider returned server error ({status_code}). "
                "The service may be experiencing issues. Try again later."
            )

    # API key errors (various formats)
    if "api key" in error_str or "apikey" in error_str or "unauthorized" in error_str:
        return (
            "LLM authentication error",
            "API key issue. Check that LLM_API_KEY is set correctly in your .env file."
        )

    # Model not found
    if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
        return (
            "LLM model not found",
            "The specified model is not available. Check that LLM_MODEL is correct "
            "and available from your provider."
        )

    # SSL/TLS errors
    if "ssl" in error_str or "certificate" in error_str:
        return (
            "LLM SSL error",
            "SSL/TLS error connecting to LLM provider. Check that LLM_BASE_URL uses "
            "the correct protocol (http vs https)."
        )

    # DNS resolution failures
    if "name resolution" in error_str or "getaddrinfo" in error_str or "nodename" in error_str:
        return (
            "LLM host not found",
            "Cannot resolve LLM provider hostname. Check that LLM_BASE_URL is correct "
            "and you have internet connectivity."
        )

    # Generic connection errors
    if "connect" in error_str or "connection" in error_str:
        return (
            "LLM connection error",
            f"Connection error: {error}. Check your network and LLM_BASE_URL setting."
        )

    # Fallback for unknown errors
    logger.warning(f"Unclassified LLM error: {error_type}: {error}")
    return (
        "LLM error",
        f"Error communicating with LLM: {error}"
    )


def format_error_for_cli(error: Exception) -> str:
    """
    Format an error for CLI display with Rich markup.

    Args:
        error: The exception that was raised

    Returns:
        Formatted error string with Rich markup
    """
    short_msg, detailed_msg = classify_llm_error(error)
    return f"[red bold]{short_msg}[/red bold]\n[dim]{detailed_msg}[/dim]"


def format_error_for_slack(error: Exception) -> str:
    """
    Format an error for Slack display.

    Args:
        error: The exception that was raised

    Returns:
        Formatted error string for Slack
    """
    short_msg, detailed_msg = classify_llm_error(error)
    return f"⚠️ *{short_msg}*\n{detailed_msg}"


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is likely temporary and worth retrying.

    Args:
        error: The exception that was raised

    Returns:
        True if the error might succeed on retry
    """
    error_str = str(error).lower()

    # Timeouts are often temporary
    if isinstance(error, (ConnectTimeout, ReadTimeout)):
        return True

    # Rate limiting is temporary
    if isinstance(error, HTTPStatusError) and error.response.status_code == 429:
        return True

    # Server errors may be temporary
    if isinstance(error, HTTPStatusError) and error.response.status_code >= 500:
        return True

    # Connection issues may be temporary
    if "timeout" in error_str or "timed out" in error_str:
        return True

    return False
