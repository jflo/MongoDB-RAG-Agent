"""Tests for think block filtering in CLI streaming."""

import pytest
from src.response_filter import (
    filter_think_streaming,
    filter_think_content,
    filter_response,
    markdown_to_slack,
    filter_response_for_slack
)


class TestThinkBlockFilter:
    """Test cases for filter_think_streaming function."""

    def test_normal_text_no_think_block(self):
        """Normal text without any think block should be output after buffer threshold."""
        # Simulate streaming normal text in chunks
        buffer = ""
        state = "buffering"
        output_parts = []

        chunks = ["Hello, ", "this is ", "a normal ", "response ", "without ", "any ", "think ", "blocks."]

        for chunk in chunks:
            filtered, buffer, state = filter_think_streaming(chunk, buffer, state)
            if filtered:
                output_parts.append(filtered)

        # Flush remaining buffer
        if buffer:
            output_parts.append(buffer)

        result = "".join(output_parts)
        assert result == "Hello, this is a normal response without any think blocks."

    def test_explicit_think_block(self):
        """<think>...</think> block should be filtered out."""
        buffer = ""
        state = "buffering"
        output_parts = []

        # Simulate chunks with explicit think tags
        chunks = [
            "<think>",
            "Let me think about this...",
            " I should search for information.",
            "</think>",
            "Based on my analysis, here is the answer."
        ]

        for chunk in chunks:
            filtered, buffer, state = filter_think_streaming(chunk, buffer, state)
            if filtered:
                output_parts.append(filtered)

        if buffer:
            output_parts.append(buffer)

        result = "".join(output_parts)
        assert result == "Based on my analysis, here is the answer."
        assert "<think>" not in result
        assert "</think>" not in result

    def test_missing_opening_think_tag(self):
        """Think content without <think> but with </think> should be filtered."""
        buffer = ""
        state = "buffering"
        output_parts = []

        # Model outputs thinking without opening tag
        chunks = [
            "Let me analyze this question...",
            " I need to search for relevant documents.",
            " The user wants to know about X.",
            "</think>",
            "Here is what I found about X."
        ]

        for chunk in chunks:
            filtered, buffer, state = filter_think_streaming(chunk, buffer, state)
            if filtered:
                output_parts.append(filtered)

        if buffer:
            output_parts.append(buffer)

        result = "".join(output_parts)
        assert result == "Here is what I found about X."
        assert "analyze" not in result
        assert "</think>" not in result

    def test_think_tag_split_across_chunks(self):
        """</think> tag split across chunks should still be detected."""
        buffer = ""
        state = "buffering"
        output_parts = []

        # Tag split across chunks
        chunks = [
            "Thinking content here...",
            "</thi",
            "nk>",
            "Actual response."
        ]

        for chunk in chunks:
            filtered, buffer, state = filter_think_streaming(chunk, buffer, state)
            if filtered:
                output_parts.append(filtered)

        if buffer:
            output_parts.append(buffer)

        result = "".join(output_parts)
        assert result == "Actual response."

    def test_partial_tag_at_end_keeps_buffering(self):
        """Partial </think> at end should keep buffering."""
        chunk = "Some content</th"
        filtered, buffer, state = filter_think_streaming(chunk, "", "buffering")

        # Should keep buffering due to potential partial tag
        assert state == "buffering"
        assert buffer == chunk
        assert filtered == ""

    def test_long_content_without_think_stays_buffering(self):
        """Content without </think> stays buffering until stream ends."""
        long_content = "A" * 600
        filtered, buffer, state = filter_think_streaming(long_content, "", "buffering")

        # Should keep buffering - flush happens at end of stream
        assert state == "buffering"
        assert filtered == ""
        assert buffer == long_content

    def test_normal_state_passes_through(self):
        """Once in normal state, all content passes through."""
        filtered, buffer, state = filter_think_streaming("any content", "", "normal")

        assert state == "normal"
        assert filtered == "any content"
        assert buffer == ""

    def test_newline_after_think(self):
        """Newline immediately after </think> should be preserved."""
        buffer = ""
        state = "buffering"
        output_parts = []

        chunks = [
            "Thinking...",
            "</think>\n",
            "Response on new line."
        ]

        for chunk in chunks:
            filtered, buffer, state = filter_think_streaming(chunk, buffer, state)
            if filtered:
                output_parts.append(filtered)

        if buffer:
            output_parts.append(buffer)

        result = "".join(output_parts)
        assert result == "\nResponse on new line."

    def test_realistic_streaming_scenario(self):
        """Simulate realistic small chunks from streaming API."""
        buffer = ""
        state = "buffering"
        output_parts = []

        # Realistic small chunks like from OpenAI streaming
        chunks = [
            "I",
            " need",
            " to",
            " think",
            " about",
            " this",
            " query",
            ".",
            " Let",
            " me",
            " search",
            ".",
            "</",
            "think",
            ">",
            "\n\n",
            "Based",
            " on",
            " the",
            " documents",
            ","
        ]

        for chunk in chunks:
            filtered, buffer, state = filter_think_streaming(chunk, buffer, state)
            if filtered:
                output_parts.append(filtered)

        if buffer:
            output_parts.append(buffer)

        result = "".join(output_parts)
        assert "I need to think" not in result
        assert "</think>" not in result
        assert "Based on the documents," in result


class TestEdgeCases:
    """Edge cases and potential failure scenarios."""

    def test_empty_chunk(self):
        """Empty chunk should not break the filter."""
        filtered, buffer, state = filter_think_streaming("", "existing", "buffering")
        assert buffer == "existing"
        assert state == "buffering"

    def test_only_think_content(self):
        """Response that is only think content should result in empty output."""
        buffer = ""
        state = "buffering"
        output_parts = []

        chunks = ["<think>", "Only thinking, no response", "</think>"]

        for chunk in chunks:
            filtered, buffer, state = filter_think_streaming(chunk, buffer, state)
            if filtered:
                output_parts.append(filtered)

        if buffer:
            output_parts.append(buffer)

        result = "".join(output_parts)
        assert result == ""

    def test_multiple_think_blocks(self):
        """Multiple think blocks - only first </think> matters."""
        buffer = ""
        state = "buffering"
        output_parts = []

        # Second think block shouldn't be filtered after first closes
        chunks = [
            "<think>First thought</think>",
            "Response ",
            "<think>This shows because we're in normal state</think>"
        ]

        for chunk in chunks:
            filtered, buffer, state = filter_think_streaming(chunk, buffer, state)
            if filtered:
                output_parts.append(filtered)

        if buffer:
            output_parts.append(buffer)

        result = "".join(output_parts)
        # After first </think>, we're in normal mode - everything passes through
        assert "Response " in result

    def test_angle_bracket_in_normal_text(self):
        """Angle brackets in normal text shouldn't cause issues once in normal state."""
        # In normal state, angle brackets pass through
        filtered, buffer, state = filter_think_streaming(" x < y and y > z", "", "normal")
        assert filtered == " x < y and y > z"
        assert state == "normal"

    def test_long_think_content_before_close_tag(self):
        """Think content > 500 chars before </think> - THIS IS THE BUG!

        If the model produces more than 500 chars of thinking before
        the </think> tag, the current logic will output it prematurely.
        """
        buffer = ""
        state = "buffering"
        output_parts = []

        # Long think content split into chunks
        think_content = "A" * 600  # More than 500 char threshold
        chunks = [think_content, "</think>", "Actual response"]

        for chunk in chunks:
            filtered, buffer, state = filter_think_streaming(chunk, buffer, state)
            if filtered:
                output_parts.append(filtered)

        if buffer:
            output_parts.append(buffer)

        result = "".join(output_parts)
        # This test will FAIL with current implementation -
        # it will output the 600 A's before seeing </think>
        assert result == "Actual response", f"Got: {result[:100]}..."
        assert "AAAA" not in result


class TestToolArtifactFilter:
    """Test cases for tool artifact filtering."""

    def test_filter_response_removes_think_and_tools(self):
        """filter_response should remove both think blocks and tool artifacts."""
        text = '<think>Let me think</think>Here is the answer.'
        result = filter_response(text)
        assert result == "Here is the answer."

    def test_filter_tool_json_artifacts(self):
        """JSON tool call syntax should be removed."""
        text = 'I will search. {"tool": "search", "args": {"query": "test"}} Found results.'
        result = filter_response(text)
        assert '{"tool"' not in result
        assert "Found results" in result

    def test_filter_tool_result_markers(self):
        """Tool result markers should be removed."""
        text = 'Searching... [Tool Result: Found 5 documents] Here are the results.'
        result = filter_response(text)
        assert "[Tool Result" not in result
        assert "Here are the results" in result

    def test_clean_text_unchanged(self):
        """Clean text should pass through unchanged."""
        text = "This is a normal response with no special content."
        result = filter_response(text)
        assert result == text

    def test_batch_think_filter(self):
        """filter_think_content should work on complete text."""
        text = "<think>Internal thoughts</think>External response."
        result = filter_think_content(text)
        assert result == "External response."
        assert "Internal" not in result


class TestMarkdownToSlack:
    """Test cases for Markdown to Slack mrkdwn conversion."""

    def test_bold_double_asterisk(self):
        """**bold** should become *bold*."""
        result = markdown_to_slack("This is **bold** text.")
        assert result == "This is *bold* text."

    def test_bold_double_underscore(self):
        """__bold__ should become *bold*."""
        result = markdown_to_slack("This is __bold__ text.")
        assert result == "This is *bold* text."

    def test_italic_single_asterisk(self):
        """*italic* should become _italic_."""
        result = markdown_to_slack("This is *italic* text.")
        assert result == "This is _italic_ text."

    def test_strikethrough(self):
        """~~strikethrough~~ should become ~strikethrough~."""
        result = markdown_to_slack("This is ~~deleted~~ text.")
        assert result == "This is ~deleted~ text."

    def test_link_conversion(self):
        """[text](url) should become <url|text>."""
        result = markdown_to_slack("Check [this link](https://example.com) out.")
        assert result == "Check <https://example.com|this link> out."

    def test_header_h1(self):
        """# Header should become *Header*."""
        result = markdown_to_slack("# Main Title\nSome content.")
        assert result == "*Main Title*\nSome content."

    def test_header_h2(self):
        """## Header should become *Header*."""
        result = markdown_to_slack("## Section Title")
        assert result == "*Section Title*"

    def test_header_h3(self):
        """### Header should become *Header*."""
        result = markdown_to_slack("### Subsection")
        assert result == "*Subsection*"

    def test_bullet_list_dash(self):
        """- item should become • item."""
        result = markdown_to_slack("- First item\n- Second item")
        assert result == "• First item\n• Second item"

    def test_bullet_list_asterisk(self):
        """* item should become • item."""
        result = markdown_to_slack("* First item\n* Second item")
        assert result == "• First item\n• Second item"

    def test_inline_code_preserved(self):
        """`code` should remain unchanged."""
        result = markdown_to_slack("Use the `print()` function.")
        assert result == "Use the `print()` function."

    def test_code_block_preserved(self):
        """```code``` blocks should remain unchanged."""
        text = "Example:\n```python\nprint('hello')\n```\nDone."
        result = markdown_to_slack(text)
        assert "```python\nprint('hello')\n```" in result

    def test_code_block_not_affected_by_bold(self):
        """Bold syntax inside code blocks should not be converted."""
        text = "```\n**not bold**\n```"
        result = markdown_to_slack(text)
        assert "**not bold**" in result

    def test_inline_code_not_affected_by_italic(self):
        """Italic syntax inside inline code should not be converted."""
        result = markdown_to_slack("Use `*args` for variable arguments.")
        assert "`*args`" in result

    def test_complex_formatting(self):
        """Multiple formatting types in one message."""
        text = "# Title\n\nThis is **bold** and *italic* text.\n\n- Item 1\n- Item 2\n\nSee [docs](https://docs.com)."
        result = markdown_to_slack(text)
        assert "*Title*" in result
        assert "*bold*" in result
        assert "_italic_" in result
        assert "• Item 1" in result
        assert "<https://docs.com|docs>" in result

    def test_empty_string(self):
        """Empty string should return empty string."""
        result = markdown_to_slack("")
        assert result == ""

    def test_plain_text_unchanged(self):
        """Plain text without formatting should remain unchanged."""
        text = "Just plain text with no special formatting."
        result = markdown_to_slack(text)
        assert result == text


class TestFilterResponseForSlack:
    """Test cases for combined Slack response filtering."""

    def test_filters_think_and_converts_markdown(self):
        """Should filter think blocks and convert Markdown."""
        text = "<think>Internal thought</think>Here is **bold** text."
        result = filter_response_for_slack(text)
        assert "Internal thought" not in result
        assert "*bold*" in result
        assert "**bold**" not in result

    def test_filters_tools_and_converts_markdown(self):
        """Should filter tool artifacts and convert Markdown."""
        text = '{"tool": "search", "args": {}} Check [this](https://example.com).'
        result = filter_response_for_slack(text)
        assert '{"tool"' not in result
        assert "<https://example.com|this>" in result

    def test_full_pipeline(self):
        """Full filtering and conversion pipeline."""
        text = """<think>Let me think about this...</think>
# Answer

Here is the **answer** to your question:

- Point 1
- Point 2

See [documentation](https://docs.example.com) for more."""
        result = filter_response_for_slack(text)
        assert "Let me think" not in result
        assert "*Answer*" in result
        assert "*answer*" in result
        assert "• Point 1" in result
        assert "<https://docs.example.com|documentation>" in result
