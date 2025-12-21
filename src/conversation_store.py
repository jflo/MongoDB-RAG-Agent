"""MongoDB-backed conversation history for Slack bot."""

import logging
from typing import List, Optional, Any
from datetime import datetime, timezone

from pydantic_ai.messages import ModelMessage

logger = logging.getLogger(__name__)


class ConversationStore:
    """Store and retrieve conversation history per channel/user."""

    def __init__(self, collection: Any):
        """
        Initialize conversation store.

        Args:
            collection: MongoDB collection for conversations
        """
        self.collection = collection

    async def get_history(
        self,
        channel_id: str,
        user_id: str,
        limit: int = 20
    ) -> List[ModelMessage]:
        """
        Retrieve recent conversation history.

        Args:
            channel_id: Slack channel ID
            user_id: Slack user ID
            limit: Maximum number of message pairs to return

        Returns:
            List of ModelMessage objects for Pydantic AI
        """
        try:
            doc = await self.collection.find_one(
                {"channel_id": channel_id, "user_id": user_id}
            )

            if not doc or "messages" not in doc:
                return []

            # Deserialize stored messages back to ModelMessage objects
            messages: List[ModelMessage] = []
            stored_messages = doc["messages"][-limit * 2:]  # Get last N pairs

            for msg_data in stored_messages:
                try:
                    # Pydantic AI messages use TypeAdapter for polymorphic deserialization
                    from pydantic_ai.messages import ModelMessagesTypeAdapter
                    parsed = ModelMessagesTypeAdapter.validate_python([msg_data])
                    messages.extend(parsed)
                except Exception as e:
                    logger.warning(f"Failed to deserialize message: {e}")
                    continue

            logger.debug(f"Retrieved {len(messages)} messages for {channel_id}/{user_id}")
            return messages

        except Exception as e:
            logger.exception(f"Error retrieving conversation history: {e}")
            return []

    async def save_messages(
        self,
        channel_id: str,
        user_id: str,
        messages: List[ModelMessage]
    ) -> None:
        """
        Save new messages to conversation history.

        Args:
            channel_id: Slack channel ID
            user_id: Slack user ID
            messages: List of ModelMessage objects from Pydantic AI
        """
        if not messages:
            return

        try:
            # Serialize messages using Pydantic AI's TypeAdapter
            from pydantic_ai.messages import ModelMessagesTypeAdapter
            serialized: List[Any] = []

            for msg in messages:
                # Serialize each message to dict
                msg_dict = ModelMessagesTypeAdapter.dump_python([msg])[0]
                serialized.append(msg_dict)

            now = datetime.now(timezone.utc)

            # Upsert the conversation document
            result = await self.collection.update_one(
                {"channel_id": channel_id, "user_id": user_id},
                {
                    "$push": {"messages": {"$each": serialized}},
                    "$set": {"updated_at": now},
                    "$setOnInsert": {"created_at": now}
                },
                upsert=True
            )

            logger.debug(
                f"Saved {len(messages)} messages for {channel_id}/{user_id}, "
                f"modified={result.modified_count}"
            )

        except Exception as e:
            logger.exception(f"Error saving conversation history: {e}")

    async def clear_history(
        self,
        channel_id: str,
        user_id: Optional[str] = None
    ) -> int:
        """
        Clear conversation history.

        Args:
            channel_id: Slack channel ID
            user_id: Optional user ID (if None, clears all users in channel)

        Returns:
            Number of documents deleted
        """
        try:
            query = {"channel_id": channel_id}
            if user_id:
                query["user_id"] = user_id

            result = await self.collection.delete_many(query)
            logger.info(
                f"Cleared {result.deleted_count} conversations for {channel_id}"
                + (f"/{user_id}" if user_id else "")
            )
            return result.deleted_count

        except Exception as e:
            logger.exception(f"Error clearing conversation history: {e}")
            return 0

    async def trim_history(
        self,
        channel_id: str,
        user_id: str,
        keep_count: int = 50
    ) -> None:
        """
        Trim conversation history to prevent unbounded growth.

        Args:
            channel_id: Slack channel ID
            user_id: Slack user ID
            keep_count: Number of recent messages to keep
        """
        try:
            doc = await self.collection.find_one(
                {"channel_id": channel_id, "user_id": user_id}
            )

            if not doc or "messages" not in doc:
                return

            messages = doc["messages"]
            if len(messages) <= keep_count:
                return

            # Keep only the most recent messages
            trimmed = messages[-keep_count:]
            await self.collection.update_one(
                {"channel_id": channel_id, "user_id": user_id},
                {"$set": {"messages": trimmed, "updated_at": datetime.now(timezone.utc)}}
            )

            logger.info(
                f"Trimmed conversation for {channel_id}/{user_id}: "
                f"{len(messages)} -> {len(trimmed)} messages"
            )

        except Exception as e:
            logger.exception(f"Error trimming conversation history: {e}")
