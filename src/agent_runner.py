"""Shared agent interaction logic for CLI and Slack interfaces.

Provides both streaming and non-streaming execution modes with proper
error handling and response filtering.
"""

import logging
from dataclasses import dataclass
from typing import AsyncIterator, Callable, List, Optional

from pydantic_ai import Agent
from pydantic_ai.messages import (
    PartDeltaEvent, PartStartEvent, TextPartDelta,
    ModelRequest, SystemPromptPart
)
from pydantic_ai.ag_ui import StateDeps

from src.agent import rag_agent, RAGState
from src.prompts import MAIN_SYSTEM_PROMPT
from src.response_filter import filter_response, filter_think_streaming
from src.errors import format_error_for_cli, format_error_for_slack, is_retryable_error

logger = logging.getLogger(__name__)


def _log_system_prompt() -> None:
    """Log the system prompt being used (first 200 chars for brevity)."""
    preview = MAIN_SYSTEM_PROMPT[:200].replace('\n', ' ')
    logger.debug(f"System prompt preview: {preview}...")


def _strip_system_prompts(message_history: List) -> List:
    """
    Remove SystemPromptPart from message history.

    When message_history contains a SystemPromptPart, Pydantic AI uses that
    instead of the agent's configured system_prompt. We strip it so the
    agent's current system prompt is always used.

    Args:
        message_history: List of ModelMessage objects

    Returns:
        List with SystemPromptParts removed from ModelRequests
    """
    if not message_history:
        return message_history

    cleaned = []
    for msg in message_history:
        if isinstance(msg, ModelRequest):
            # Filter out SystemPromptPart from this request's parts
            filtered_parts = [
                part for part in msg.parts
                if not isinstance(part, SystemPromptPart)
            ]
            if filtered_parts:
                # Create new ModelRequest with filtered parts
                cleaned.append(ModelRequest(parts=filtered_parts))
            # If no parts remain, skip this message entirely
        else:
            cleaned.append(msg)

    return cleaned


@dataclass
class AgentResult:
    """Result from agent execution."""

    response: str
    """Filtered response text."""

    new_messages: List
    """New messages to add to conversation history."""

    error: Optional[str] = None
    """Error message if execution failed."""


async def run_agent(
    user_input: str,
    deps: StateDeps[RAGState],
    message_history: Optional[List] = None,
) -> AgentResult:
    """
    Run agent without streaming (for Slack and batch processing).

    Args:
        user_input: The user's query text
        deps: StateDeps with RAG state
        message_history: Optional conversation history

    Returns:
        AgentResult with response and new messages
    """
    message_history = message_history or []

    # Strip any existing system prompts from history so the agent's
    # configured system_prompt is always used (not stale ones from DB)
    cleaned_history = _strip_system_prompts(message_history)

    # Debug logging
    _log_system_prompt()
    logger.info(f"Running agent with input: {user_input[:100]}...")
    logger.debug(f"Message history: {len(message_history)} -> {len(cleaned_history)} after stripping system prompts")

    try:
        result = await rag_agent.run(
            user_input,
            deps=deps,
            message_history=cleaned_history
        )

        # Get and filter response
        response = result.output if hasattr(result, "output") else str(result)
        response = filter_response(response)

        if not response:
            response = "I processed your request but have no response to share."

        return AgentResult(
            response=response,
            new_messages=result.new_messages()
        )

    except Exception as e:
        logger.exception(f"Agent execution failed: {e}")
        error_msg = format_error_for_slack(e)

        if is_retryable_error(e):
            error_msg += "\n\n_This error may be temporary. Please try again in a moment._"

        return AgentResult(
            response="",
            new_messages=[],
            error=error_msg
        )


@dataclass
class StreamChunk:
    """A chunk of streamed content."""

    text: str
    """Text content to display."""

    is_final: bool = False
    """Whether this is the final chunk."""


async def stream_agent(
    user_input: str,
    deps: StateDeps[RAGState],
    message_history: Optional[List] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
) -> AgentResult:
    """
    Stream agent execution with real-time output.

    Args:
        user_input: The user's query text
        deps: StateDeps with RAG state
        message_history: Optional conversation history
        on_chunk: Optional callback for each text chunk (for custom output)

    Returns:
        AgentResult with full response and new messages
    """
    message_history = message_history or []

    # Strip any existing system prompts from history so the agent's
    # configured system_prompt is always used (not stale ones from DB)
    cleaned_history = _strip_system_prompts(message_history)

    # Debug logging
    _log_system_prompt()
    logger.info(f"Streaming agent with input: {user_input[:100]}...")
    logger.debug(f"Message history: {len(message_history)} -> {len(cleaned_history)} after stripping system prompts")

    try:
        response_text = ""
        think_buffer = ""
        think_state = "buffering"

        async with rag_agent.iter(
            user_input,
            deps=deps,
            message_history=cleaned_history
        ) as run:

            async for node in run:

                if Agent.is_user_prompt_node(node):
                    pass  # No action needed

                elif Agent.is_model_request_node(node):
                    # Reset think state for each new model request
                    think_buffer = ""
                    think_state = "buffering"

                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            # Handle text part start events
                            if isinstance(event, PartStartEvent) and event.part.part_kind == 'text':
                                initial_text = event.part.content
                                if initial_text:
                                    filtered, think_buffer, think_state = filter_think_streaming(
                                        initial_text, think_buffer, think_state
                                    )
                                    if filtered and on_chunk:
                                        on_chunk(filtered)
                                    response_text += initial_text

                            # Handle text delta events
                            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                delta_text = event.delta.content_delta
                                if delta_text:
                                    filtered, think_buffer, think_state = filter_think_streaming(
                                        delta_text, think_buffer, think_state
                                    )
                                    if filtered and on_chunk:
                                        on_chunk(filtered)
                                    response_text += delta_text

                    # Flush remaining buffer if no </think> found
                    if think_buffer and think_state == "buffering":
                        if on_chunk:
                            on_chunk(think_buffer)
                        think_buffer = ""
                        think_state = "normal"

                elif Agent.is_call_tools_node(node):
                    # Execute tool calls silently
                    async with node.stream(run.ctx) as tool_stream:
                        async for _ in tool_stream:
                            pass

                elif Agent.is_end_node(node):
                    pass

        # Get final output
        final_output = run.result.output if hasattr(run.result, 'output') else str(run.result)
        response = response_text.strip() or final_output

        return AgentResult(
            response=response,
            new_messages=run.result.new_messages()
        )

    except Exception as e:
        logger.exception(f"Agent streaming failed: {e}")
        error_msg = format_error_for_cli(e)

        return AgentResult(
            response="",
            new_messages=[],
            error=error_msg
        )
