#!/usr/bin/env python3
"""Slack bot interface for MongoDB RAG Agent using Socket Mode."""

import asyncio
import logging
import re
import sys

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from pydantic_ai.ag_ui import StateDeps

from src.agent import RAGState
from src.settings import load_settings
from src.agent_runner import run_agent
from src.komga import get_komga_client
from src.conversation_store import ConversationStore
from src.response_filter import filter_response_for_slack
from src.errors import format_error_for_slack, is_retryable_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load settings
settings = load_settings()

# Log LLM configuration for debugging
logger.info(f"LLM configured: model={settings.llm_model}, base_url={settings.llm_base_url}")

# Validate Slack configuration
if not settings.slack_bot_token or not settings.slack_app_token:
    logger.error(
        "Slack tokens not configured. Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN in .env"
    )
    sys.exit(1)

# Initialize Slack app
app = AsyncApp(
    token=settings.slack_bot_token,
    signing_secret=settings.slack_signing_secret or None
)


def _extract_query(text: str) -> str:
    """
    Remove bot mention from message text.

    Args:
        text: Raw message text from Slack (e.g., "<@U123BOT> what is X?")

    Returns:
        Query text without mention (e.g., "what is X?")
    """
    # Remove <@UXXXXX> mention patterns
    return re.sub(r"<@U[A-Z0-9]+>", "", text).strip()


def _markdown_to_slack_mrkdwn(text: str) -> str:
    """
    Convert markdown links to Slack mrkdwn format.

    Handles both standard markdown [text](url) and fullwidth unicode variants
    that LLMs sometimes produce: 【text】(url) or 【[text]url】

    Args:
        text: Text with markdown links

    Returns:
        Text with Slack mrkdwn links like <url|text>
    """
    # Standard markdown links [text](url) to Slack format <url|text>
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', text)

    # Fullwidth bracket variant: 【[text]url】 (LLM sometimes produces this)
    text = re.sub(r'【\[([^\]]+)\](https?://[^】]+)】', r'<\2|\1>', text)

    # Fullwidth bracket variant: 【text】(url)
    text = re.sub(r'【([^】]+)】\(([^)]+)\)', r'<\2|\1>', text)

    return text


@app.event("app_mention")
async def handle_mention(event: dict, say, client) -> None:
    """
    Handle @mentions of the bot in channels.

    Args:
        event: Slack event payload
        say: Function to send messages to channel
        client: Slack WebClient for API calls
    """
    channel_id = event["channel"]
    user_id = event["user"]
    text = event["text"]

    logger.info(f"Received mention from {user_id}: {text[:50]}...")

    # Extract query from mention
    query = _extract_query(text)

    if not query:
        await say(
            text="Hello! Ask me anything about the knowledge base."
        )
        return

    # Post a "thinking" message to show we're processing
    thinking_msg = await client.chat_postMessage(
        channel=channel_id,
        text="_Thinking..._"
    )
    thinking_ts = thinking_msg["ts"]

    try:
        # Create agent state and deps
        state = RAGState()
        deps = StateDeps[RAGState](state=state)

        # Run agent with no message history (stateless)
        result = await run_agent(
            user_input=query,
            deps=deps,
            message_history=[]
        )

        # Delete the thinking message
        await client.chat_delete(channel=channel_id, ts=thinking_ts)
        logger.info("LLM agent call completed successfully")

        # Get and clean response (filter think blocks, tool artifacts, and linkify citations)
        response = filter_response_for_slack(result.response, state.citation_map)

        # Handle errors
        if result.error:
            await say(text=result.error)
            return

        # Convert markdown links to Slack mrkdwn format and send
        slack_text = _markdown_to_slack_mrkdwn(response)
        await say(text=slack_text)

        logger.info(f"Replied to {user_id}")

    except Exception as e:
        # Try to delete thinking message on error
        try:
            await client.chat_delete(channel=channel_id, ts=thinking_ts)
        except Exception:
            pass
        raise


@app.event("message")
async def handle_message(event: dict) -> None:
    """
    Handle direct messages to the bot.

    Currently a no-op - we only respond to mentions.
    This handler prevents warnings about unhandled events.
    """
    # Log all message events for debugging
    subtype = event.get("subtype", "normal")
    text = event.get("text", "")[:50] if event.get("text") else "(no text)"
    logger.debug(f"Message event received: subtype={subtype}, text={text}...")
    pass


async def main() -> None:
    """Run the Slack bot."""
    logger.info("=" * 50)
    logger.info("MongoDB RAG Agent - Slack Bot")
    logger.info("=" * 50)

    # Test Komga connectivity if configured
    komga = get_komga_client(settings)
    if komga.is_configured():
        success, message = await komga.test_connection()
        if success:
            logger.info(f"Komga: {message}")
        else:
            logger.warning(f"Komga: {message}")
    else:
        logger.info("Komga: Not configured (PDF deep links disabled)")

    # Start Socket Mode handler
    logger.info("Starting Slack bot in Socket Mode...")
    logger.info("Make sure your Slack app has:")
    logger.info("  - Event Subscriptions enabled with 'app_mention' event")
    logger.info("  - Bot scopes: app_mentions:read, chat:write")
    logger.info("Waiting for events... (Ctrl+C to quit)")

    handler = AsyncSocketModeHandler(app, settings.slack_app_token)
    try:
        await handler.start_async()
    except asyncio.CancelledError:
        # Raised when Ctrl+C is pressed - asyncio.run() cancels all tasks
        pass
    finally:
        logger.info("Shutting down...")
        await handler.close_async()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Suppress the KeyboardInterrupt traceback on clean shutdown
        pass
