#!/usr/bin/env python3
"""Slack bot interface for MongoDB RAG Agent using Socket Mode."""

import asyncio
import logging
import re
import sys

from pymongo import AsyncMongoClient
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from pydantic_ai.ag_ui import StateDeps

from src.agent import rag_agent, RAGState
from src.settings import load_settings
from src.conversation_store import ConversationStore

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

# Global resources (initialized on startup)
mongo_client: AsyncMongoClient = None
conversation_store: ConversationStore = None


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


def _filter_think_content(text: str) -> str:
    """
    Filter out think blocks from response.

    Args:
        text: Raw response that may contain <think>...</think> blocks

    Returns:
        Text with think blocks removed
    """
    # Remove think blocks including newlines after
    filtered = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    # Also handle case where there's no opening tag but closing exists
    if "</think>" in filtered:
        # Find closing tag and remove everything before it
        idx = filtered.find("</think>")
        filtered = filtered[idx + len("</think>"):].strip()
    return filtered.strip()


@app.event("app_mention")
async def handle_mention(event: dict, say) -> None:
    """
    Handle @mentions of the bot in channels.

    Args:
        event: Slack event payload
        say: Function to send messages to channel
    """
    channel_id = event["channel"]
    user_id = event["user"]
    text = event["text"]
    thread_ts = event.get("thread_ts") or event["ts"]

    logger.info(f"Received mention from {user_id} in {channel_id}: {text[:50]}...")

    # Extract query from mention
    query = _extract_query(text)

    if not query:
        await say(
            text="Hello! Ask me anything about the knowledge base.",
            thread_ts=thread_ts
        )
        return

    try:
        # Get conversation history for context
        message_history = await conversation_store.get_history(channel_id, user_id)
        logger.debug(f"Retrieved {len(message_history)} messages from history")

        # Create agent state and deps
        state = RAGState()
        deps = StateDeps[RAGState](state=state)

        # Run agent (non-streaming for Slack)
        result = await rag_agent.run(
            query,
            deps=deps,
            message_history=message_history
        )

        # Get and clean response
        response = result.output if hasattr(result, "output") else str(result)
        response = _filter_think_content(response)

        if not response:
            response = "I processed your request but have no response to share."

        # Save updated conversation history
        new_messages = result.new_messages()
        await conversation_store.save_messages(channel_id, user_id, new_messages)

        # Periodically trim history to prevent unbounded growth
        await conversation_store.trim_history(channel_id, user_id, keep_count=50)

        # Reply in thread
        await say(text=response, thread_ts=thread_ts)
        logger.info(f"Replied to {user_id} in thread {thread_ts}")

    except Exception as e:
        logger.exception(f"Error processing Slack message: {e}")
        await say(
            text=f"Sorry, I encountered an error: {str(e)}",
            thread_ts=thread_ts
        )


@app.event("message")
async def handle_message(event: dict) -> None:
    """
    Handle direct messages to the bot.

    Currently a no-op - we only respond to mentions.
    This handler prevents warnings about unhandled events.
    """
    # Only respond to mentions, not all messages
    pass


async def initialize_resources() -> None:
    """Initialize MongoDB connection and conversation store."""
    global mongo_client, conversation_store

    logger.info("Initializing MongoDB connection...")

    mongo_client = AsyncMongoClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5000
    )

    # Verify connection
    await mongo_client.admin.command("ping")
    logger.info(f"Connected to MongoDB: {settings.mongodb_database}")

    # Initialize conversation store
    db = mongo_client[settings.mongodb_database]
    collection = db[settings.mongodb_collection_conversations]
    conversation_store = ConversationStore(collection)

    # Create indexes for efficient queries
    await collection.create_index([
        ("channel_id", 1),
        ("user_id", 1)
    ], unique=True)
    await collection.create_index("updated_at")

    logger.info("Conversation store initialized")


async def cleanup_resources() -> None:
    """Clean up resources on shutdown."""
    global mongo_client

    if mongo_client:
        await mongo_client.close()
        logger.info("MongoDB connection closed")


async def main() -> None:
    """Run the Slack bot."""
    logger.info("=" * 50)
    logger.info("MongoDB RAG Agent - Slack Bot")
    logger.info("=" * 50)

    try:
        # Initialize resources
        await initialize_resources()

        # Start Socket Mode handler
        logger.info("Starting Slack bot in Socket Mode...")
        handler = AsyncSocketModeHandler(app, settings.slack_app_token)
        await handler.start_async()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise
    finally:
        await cleanup_resources()


if __name__ == "__main__":
    asyncio.run(main())
