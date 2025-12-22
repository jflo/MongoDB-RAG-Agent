"""Response filtering utilities for think blocks and tool use.

Provides both streaming and batch filtering for use by CLI and Slack interfaces.
"""

import re
from typing import Tuple


def filter_think_streaming(
    chunk: str,
    buffer: str,
    state: str
) -> Tuple[str, str, str]:
    """
    Filter think blocks during streaming.

    Strategy: Buffer all content until we see </think>. Content is only
    released when </think> is found (discarding think content) or when
    streaming ends (flush releases buffer as normal content).

    Models may omit the opening <think> tag but still include </think>.

    States:
    - "buffering": Collecting content, looking for </think>
    - "normal": Past any think block, outputting normally

    Args:
        chunk: New text chunk from stream
        buffer: Buffered text
        state: Current state ("buffering" or "normal")

    Returns:
        Tuple of (text_to_display, new_buffer, new_state)
    """
    if state == "normal":
        # Past any think block, output everything directly
        return (chunk, "", "normal")

    # state == "buffering"
    text = buffer + chunk

    # Check for </think> tag
    close_idx = text.find('</think>')
    if close_idx != -1:
        # Found closing tag - discard everything before it (think content)
        # Output everything after it
        output = text[close_idx + len('</think>'):]
        return (output, "", "normal")

    # Check for partial </think> at end that might complete in next chunk
    for suffix_len in range(1, len('</think>')):
        potential_suffix = '</think>'[:suffix_len]
        if text.endswith(potential_suffix):
            # Keep buffering - might be partial tag
            return ("", text, "buffering")

    # No </think> found - keep buffering
    # The buffer will be flushed at end of streaming if no </think> is found
    return ("", text, "buffering")


def filter_think_content(text: str) -> str:
    """
    Filter out think blocks from complete response text.

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


def filter_tool_artifacts(text: str) -> str:
    """
    Filter out tool-related artifacts from response text.

    Some models may include tool call syntax or results in their responses.
    This removes common patterns.

    Args:
        text: Response text that may contain tool artifacts

    Returns:
        Text with tool artifacts removed
    """
    # Remove JSON-like tool call blocks (e.g., {"tool": "search", "args": {...}})
    # This handles cases where the model echoes tool calls
    filtered = re.sub(
        r'\{"tool":\s*"[^"]+",\s*"args":\s*\{[^}]*\}\}',
        '',
        text
    )

    # Remove tool result markers that some models may include
    filtered = re.sub(
        r'\[Tool Result:.*?\]',
        '',
        filtered,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Remove function call syntax (e.g., search_knowledge_base(...))
    # But be careful not to remove legitimate function references in explanations
    # Only remove if it looks like actual tool output
    filtered = re.sub(
        r'search_knowledge_base\([^)]*\)\s*->\s*',
        '',
        filtered
    )

    # Clean up any double spaces or newlines left behind
    filtered = re.sub(r'\n{3,}', '\n\n', filtered)
    filtered = re.sub(r'  +', ' ', filtered)

    return filtered.strip()


def filter_response(text: str) -> str:
    """
    Apply all response filters to clean up output.

    Combines think block and tool artifact filtering.

    Args:
        text: Raw response text

    Returns:
        Cleaned response text
    """
    result = filter_think_content(text)
    result = filter_tool_artifacts(result)
    return result
