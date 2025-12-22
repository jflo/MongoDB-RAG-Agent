"""Response filtering utilities for think blocks and tool use.

Provides both streaming and batch filtering for use by CLI and Slack interfaces.
Also provides Markdown to Slack mrkdwn conversion.
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


def markdown_to_slack(text: str) -> str:
    """
    Convert Markdown formatting to Slack mrkdwn format.

    Slack uses its own markup format that differs from Markdown:
    - Bold: **text** or __text__ -> *text*
    - Italic: *text* or _text_ -> _text_
    - Strikethrough: ~~text~~ -> ~text~
    - Code: `code` -> `code` (same)
    - Code blocks: ```code``` -> ```code``` (same)
    - Links: [text](url) -> <url|text>
    - Headers: # Header -> *Header*

    Args:
        text: Text with Markdown formatting

    Returns:
        Text with Slack mrkdwn formatting
    """
    result = text

    # Protect code blocks from other transformations
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f"\x00CODE_BLOCK_{len(code_blocks) - 1}\x00"

    result = re.sub(r'```[\s\S]*?```', save_code_block, result)

    # Protect inline code
    inline_codes = []
    def save_inline_code(match):
        inline_codes.append(match.group(0))
        return f"\x00INLINE_CODE_{len(inline_codes) - 1}\x00"

    result = re.sub(r'`[^`]+`', save_inline_code, result)

    # Convert links: [text](url) -> <url|text>
    result = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', result)

    # Placeholders for bold text (to avoid italic conversion later)
    BOLD_START = "\x01BOLD_START\x01"
    BOLD_END = "\x01BOLD_END\x01"

    # Convert headers: # Header -> *Header* (bold)
    # Handle multiple header levels - use placeholder to avoid italic conversion
    def header_to_bold(match):
        return f"{BOLD_START}{match.group(1)}{BOLD_END}"
    result = re.sub(r'^#{1,6}\s+(.+)$', header_to_bold, result, flags=re.MULTILINE)

    # Convert bold: **text** or __text__ -> placeholder (to avoid italic conversion)
    def bold_to_placeholder(match):
        return f"{BOLD_START}{match.group(1)}{BOLD_END}"
    result = re.sub(r'\*\*([^*]+)\*\*', bold_to_placeholder, result)
    result = re.sub(r'__([^_]+)__', bold_to_placeholder, result)

    # Convert bullet lists BEFORE italic conversion to prevent * item -> _item_
    result = re.sub(r'^-\s+', '• ', result, flags=re.MULTILINE)
    result = re.sub(r'^\*\s+', '• ', result, flags=re.MULTILINE)

    # Convert italic: *text* -> _text_
    # Only match single asterisks not at start of line (bullets already handled)
    result = re.sub(r'(?<!\*)\*([^*\x01]+)\*(?!\*)', r'_\1_', result)

    # Convert strikethrough: ~~text~~ -> ~text~
    result = re.sub(r'~~([^~]+)~~', r'~\1~', result)

    # Restore bold placeholders to Slack bold syntax
    result = result.replace(BOLD_START, '*')
    result = result.replace(BOLD_END, '*')

    # Restore inline code
    for i, code in enumerate(inline_codes):
        result = result.replace(f"\x00INLINE_CODE_{i}\x00", code)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        result = result.replace(f"\x00CODE_BLOCK_{i}\x00", block)

    return result


def filter_response_for_slack(text: str) -> str:
    """
    Apply all response filters and convert to Slack format.

    Combines think block filtering, tool artifact filtering,
    and Markdown to Slack mrkdwn conversion.

    Args:
        text: Raw response text with Markdown

    Returns:
        Cleaned text formatted for Slack
    """
    result = filter_response(text)
    result = markdown_to_slack(result)
    return result
