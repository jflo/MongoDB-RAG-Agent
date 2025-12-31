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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load settings
settings = load_settings()

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

        # Handle errors
        if result.error:
            await say(text=result.error)
            return

        # Reply
        await say(markdown_text=result.response)
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
    # Only respond to mentions, not all messages
    pass


async def main() -> None:
    """Run the Slack bot."""
    logger.info("=" * 50)
    logger.info("MongoDB RAG Agent - Slack Bot")
    logger.info("=" * 50)

    try:
        # Start Socket Mode handler
        logger.info("Starting Slack bot in Socket Mode...")
        handler = AsyncSocketModeHandler(app, settings.slack_app_token)
        await handler.start_async()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
