#!/usr/bin/env python3
"""Conversational CLI with real-time streaming and tool call visibility."""

import asyncio
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from pydantic_ai.ag_ui import StateDeps
from dotenv import load_dotenv

from src.agent import RAGState
from src.settings import load_settings
from src.agent_runner import stream_agent
from src.komga import get_komga_client

# Load environment variables
load_dotenv(override=True)

console = Console()


async def stream_agent_interaction(
    user_input: str,
    message_history: List,
    deps: StateDeps[RAGState]
) -> tuple[str, List]:
    """
    Stream agent interaction with real-time output to Rich console.

    Args:
        user_input: The user's input text
        message_history: List of ModelRequest/ModelResponse objects for conversation context
        deps: StateDeps with RAG state

    Returns:
        Tuple of (response_text, new_messages)
    """
    # Track if we've printed the prefix yet
    prefix_printed = False

    def on_chunk(text: str) -> None:
        """Handle each streamed text chunk."""
        nonlocal prefix_printed
        if not prefix_printed:
            console.print("[bold blue]Assistant:[/bold blue] ", end="")
            prefix_printed = True
        console.print(text, end="")

    result = await stream_agent(
        user_input=user_input,
        deps=deps,
        message_history=message_history,
        on_chunk=on_chunk,
    )

    # Print newline after streaming completes
    if prefix_printed:
        console.print()

    # Handle errors
    if result.error:
        console.print(f"\n[red]{result.error}[/red]")
        console.print("[yellow]This error may be temporary. Try again in a moment.[/yellow]")
        return ("", [])

    return (result.response, result.new_messages)


def display_welcome():
    """Display welcome message with configuration info."""
    settings = load_settings()

    welcome = Panel(
        "[bold blue]MongoDB RAG Agent[/bold blue]\n\n"
        "[green]Intelligent knowledge base search with MongoDB Atlas Vector Search[/green]\n"
        f"[dim]LLM: {settings.llm_model}[/dim]\n\n"
        "[dim]Type 'exit' to quit, 'info' for system info, 'clear' to clear screen[/dim]",
        style="blue",
        padding=(1, 2)
    )
    console.print(welcome)
    console.print()


async def main():
    """Main conversation loop."""

    # Show welcome
    display_welcome()

    # Create the state that the agent will use
    state = RAGState()

    # Create StateDeps wrapper with the state
    deps = StateDeps[RAGState](state=state)

    console.print("[bold green]✓[/bold green] Search system initialized")

    # Test Komga connectivity if configured
    settings = load_settings()
    komga = get_komga_client(settings)
    if komga.is_configured():
        success, message = await komga.test_connection()
        if success:
            console.print(f"[bold green]✓[/bold green] Komga: {message}")
        else:
            console.print(f"[bold yellow]![/bold yellow] Komga: {message}")
    else:
        console.print("[dim]Komga: Not configured (PDF deep links disabled)[/dim]")

    console.print()

    # Initialize message history with proper Pydantic AI message objects
    message_history = []

    try:
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("[bold green]You").strip()

                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    console.print("\n[yellow]👋 Goodbye![/yellow]")
                    break

                elif user_input.lower() == 'info':
                    settings = load_settings()
                    console.print(Panel(
                        f"[cyan]LLM Provider:[/cyan] {settings.llm_provider}\n"
                        f"[cyan]LLM Model:[/cyan] {settings.llm_model}\n"
                        f"[cyan]Embedding Model:[/cyan] {settings.embedding_model}\n"
                        f"[cyan]Default Match Count:[/cyan] {settings.default_match_count}\n"
                        f"[cyan]Default Text Weight:[/cyan] {settings.default_text_weight}",
                        title="System Configuration",
                        border_style="magenta"
                    ))
                    continue

                elif user_input.lower() == 'clear':
                    console.clear()
                    display_welcome()
                    continue

                if not user_input:
                    continue

                # Stream the interaction and get response
                response_text, new_messages = await stream_agent_interaction(
                    user_input,
                    message_history,
                    deps
                )

                # Add new messages to history (includes both user prompt and agent response)
                message_history.extend(new_messages)

                # Add spacing after response
                console.print()

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
                continue

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                import traceback
                traceback.print_exc()
                continue

    finally:
        console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    asyncio.run(main())
