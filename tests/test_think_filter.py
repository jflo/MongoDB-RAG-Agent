"""Tests for think block filtering in CLI streaming."""

import pytest
from src.cli import _filter_think_streaming


class TestThinkBlockFilter:
    """Test cases for _filter_think_streaming function."""

    def test_normal_text_no_think_block(self):
        """Normal text without any think block should be output after buffer threshold."""
        # Simulate streaming normal text in chunks
        buffer = ""
        state = "buffering"
        output_parts = []

        chunks = ["Hello, ", "this is ", "a normal ", "response ", "without ", "any ", "think ", "blocks."]

        for chunk in chunks:
            filtered, buffer, state = _filter_think_streaming(chunk, buffer, state)
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
            filtered, buffer, state = _filter_think_streaming(chunk, buffer, state)
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
            filtered, buffer, state = _filter_think_streaming(chunk, buffer, state)
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
            filtered, buffer, state = _filter_think_streaming(chunk, buffer, state)
            if filtered:
                output_parts.append(filtered)

        if buffer:
            output_parts.append(buffer)

        result = "".join(output_parts)
        assert result == "Actual response."

    def test_partial_tag_at_end_keeps_buffering(self):
        """Partial </think> at end should keep buffering."""
        chunk = "Some content</th"
        filtered, buffer, state = _filter_think_streaming(chunk, "", "buffering")

        # Should keep buffering due to potential partial tag
        assert state == "buffering"
        assert buffer == chunk
        assert filtered == ""

    def test_long_content_without_think_stays_buffering(self):
        """Content without </think> stays buffering until stream ends."""
        long_content = "A" * 600
        filtered, buffer, state = _filter_think_streaming(long_content, "", "buffering")

        # Should keep buffering - flush happens at end of stream
        assert state == "buffering"
        assert filtered == ""
        assert buffer == long_content

    def test_normal_state_passes_through(self):
        """Once in normal state, all content passes through."""
        filtered, buffer, state = _filter_think_streaming("any content", "", "normal")

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
            filtered, buffer, state = _filter_think_streaming(chunk, buffer, state)
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
            filtered, buffer, state = _filter_think_streaming(chunk, buffer, state)
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
        filtered, buffer, state = _filter_think_streaming("", "existing", "buffering")
        assert buffer == "existing"
        assert state == "buffering"

    def test_only_think_content(self):
        """Response that is only think content should result in empty output."""
        buffer = ""
        state = "buffering"
        output_parts = []

        chunks = ["<think>", "Only thinking, no response", "</think>"]

        for chunk in chunks:
            filtered, buffer, state = _filter_think_streaming(chunk, buffer, state)
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
            filtered, buffer, state = _filter_think_streaming(chunk, buffer, state)
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
        filtered, buffer, state = _filter_think_streaming(" x < y and y > z", "", "normal")
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
            filtered, buffer, state = _filter_think_streaming(chunk, buffer, state)
            if filtered:
                output_parts.append(filtered)

        if buffer:
            output_parts.append(buffer)

        result = "".join(output_parts)
        # This test will FAIL with current implementation -
        # it will output the 600 A's before seeing </think>
        assert result == "Actual response", f"Got: {result[:100]}..."
        assert "AAAA" not in result
