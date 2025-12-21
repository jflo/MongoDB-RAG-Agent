"""Main AGUI-enabled RAG agent implementation with shared state."""

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from textwrap import dedent
import json

from pydantic_ai.ag_ui import StateDeps

from providers import get_llm_model
from dependencies import AgentDependencies
from prompts import MAIN_SYSTEM_PROMPT
from tools import semantic_search, hybrid_search


class RAGState(BaseModel):
    """Minimal shared state for the RAG agent."""
    pass


# Create the RAG agent with AGUI support
rag_agent = Agent(
    get_llm_model(),
    deps_type=StateDeps[RAGState],
    system_prompt=MAIN_SYSTEM_PROMPT
)


@rag_agent.tool
async def search_knowledge_base(
    ctx: RunContext[StateDeps[RAGState]],
    query: str,
    match_count: Optional[int] = 5,
    search_type: Optional[str] = "semantic"
) -> str:
    """
    Search the knowledge base for relevant information.

    Args:
        ctx: Agent runtime context with state dependencies
        query: Search query text
        match_count: Number of results to return (default: 5)
        search_type: Type of search - "semantic" or "hybrid" (default: semantic)

    Returns:
        String containing the retrieved information formatted for the LLM
    """
    try:
        # Initialize database connection
        agent_deps = AgentDependencies()
        await agent_deps.initialize()

        # Create a context wrapper for the search tools
        class DepsWrapper:
            def __init__(self, deps):
                self.deps = deps

        deps_ctx = DepsWrapper(agent_deps)

        # Perform the search based on type
        if search_type == "hybrid":
            results = await hybrid_search(
                ctx=deps_ctx,
                query=query,
                match_count=match_count
            )
        else:
            results = await semantic_search(
                ctx=deps_ctx,
                query=query,
                match_count=match_count
            )

        # Clean up
        await agent_deps.cleanup()

        # Format results as a simple string
        if not results:
            return "No relevant information found in the knowledge base."

        # Build a formatted response
        response_parts = [f"Found {len(results)} relevant documents:\n"]

        for i, result in enumerate(results, 1):
            # Handle both dict and object results
            if isinstance(result, dict):
                title = result.get('document_title', 'Unknown')
                content = result.get('content', '')
                similarity = result.get('combined_score', result.get('similarity', 0))
                metadata = result.get('metadata', {})
            else:
                title = result.document_title
                content = result.content
                similarity = result.similarity
                metadata = getattr(result, 'metadata', {}) or {}

            # Format page info if available
            page_info = ""
            page_numbers = metadata.get("page_numbers")
            if page_numbers:
                if len(page_numbers) == 1:
                    page_info = f", page {page_numbers[0]}"
                else:
                    page_info = f", pages {page_numbers[0]}-{page_numbers[-1]}"

            response_parts.append(f"\n--- Document {i}: {title}{page_info} (relevance: {similarity:.2f}) ---")
            response_parts.append(content)

        return "\n".join(response_parts)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

@rag_agent.instructions
async def rag_instructions(ctx: RunContext[StateDeps[RAGState]]) -> str:
    """
    Dynamic instructions for the RAG agent.

    Args:
        ctx: The run context containing RAG state information.

    Returns:
        Instructions string for the RAG agent.
    """
    return dedent(
        """
        You are an intelligent RAG (Retrieval-Augmented Generation) assistant with access to a knowledge base.

        INSTRUCTIONS:
        1. When the user asks a question, use the `search_knowledge_base` tool to find relevant information
        2. The tool will return the relevant documents and content from the knowledge base
        3. Base your answer on the retrieved information
        4. Always cite which documents you're referencing
        5. If you cannot find relevant information, be honest about it
        6. You can choose between:
           - "semantic" search for conceptual/meaning-based queries (default)
           - "hybrid" search for specific facts or keyword matching

        Be concise and helpful in your responses.
        """
    )
