"""Main MongoDB RAG agent implementation with shared state."""

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from typing import Optional, List

from pydantic_ai.ag_ui import StateDeps

from src.providers import get_llm_model
from src.dependencies import AgentDependencies
from src.prompts import MAIN_SYSTEM_PROMPT
from src.tools import semantic_search, hybrid_search, text_search, SearchResult
from src.settings import load_settings
from src.komga import get_komga_client

# Source filter patterns for document categories
RULES_FILTER = r"^GRR.*\.pdf$"  # Green Ronin rules PDFs
GAME_LOGS_FILTER = r"^GMT.*\.transcript_summary\.md$"  # Session transcripts


class RAGState(BaseModel):
    """Shared state for the RAG agent."""

    # Maps (filename, page) -> komga_url for citation linking
    # Example: {("GRR6610_TheExpanse_TUE_Core.pdf", 42): "https://komga.../read?page=42"}
    citation_map: dict[tuple[str, int], str] = {}


async def format_search_results(
    results: List[SearchResult],
    state: Optional[RAGState] = None
) -> str:
    """
    Format search results as a string for the LLM.

    Includes Komga deep links when available for PDF sources.
    Populates state.citation_map for post-processing citations.

    Args:
        results: List of search results to format
        state: Optional RAGState to populate citation_map

    Returns:
        Formatted string with document info and content
    """
    if not results:
        return "No relevant information found."

    # Get Komga client for deep linking
    settings = load_settings()
    komga = get_komga_client(settings)

    response_parts = [f"Found {len(results)} relevant documents:\n"]

    for i, result in enumerate(results, 1):
        # Format page info if available
        page_info = ""
        page_numbers = result.metadata.get("page_numbers")
        first_page = page_numbers[0] if page_numbers else None

        if page_numbers:
            if len(page_numbers) == 1:
                page_info = f", page {page_numbers[0]}"
            else:
                page_info = f", pages {page_numbers[0]}-{page_numbers[-1]}"

        # Try to get Komga deep link for PDF sources
        source_link = ""
        if result.document_source.endswith(".pdf") and komga.is_configured():
            url = await komga.get_source_url(result.document_source, first_page)
            if url:
                source_link = f" [View in Komga]({url})"
                # Populate citation map for post-processing
                if state is not None and page_numbers:
                    for page in page_numbers:
                        # Get URL for each page in the chunk
                        page_url = await komga.get_source_url(
                            result.document_source, page
                        )
                        if page_url:
                            state.citation_map[(result.document_source, page)] = (
                                page_url
                            )

        response_parts.append(
            f"\n--- Document {i}: {result.document_title} "
            f"(source: {result.document_source}){page_info}{source_link} "
            f"(relevance: {result.similarity:.2f}) ---"
        )
        response_parts.append(result.content)

    return "\n".join(response_parts)


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
    match_count: Optional[int] = 20,
    search_type: Optional[str] = "hybrid"
) -> str:
    """
    Search the knowledge base for relevant information.

    Args:
        ctx: Agent runtime context with state dependencies
        query: Search query text
        match_count: Number of results to return (default: 5)
        search_type: Type of search - "semantic" or "text" or "hybrid" (default: hybrid)

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
        elif search_type == "semantic":
            results = await semantic_search(
                ctx=deps_ctx,
                query=query,
                match_count=match_count
            )
        else:
            results = await text_search(
                ctx=deps_ctx,
                query=query,
                match_count=match_count
            )

        # Clean up
        await agent_deps.cleanup()

        # Get state for citation map population
        state = ctx.deps.state if ctx.deps else None
        return await format_search_results(results, state)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


@rag_agent.tool
async def search_rules(
    ctx: RunContext[StateDeps[RAGState]],
    query: str,
    match_count: Optional[int] = 20
) -> str:
    """
    Search the game rules and rulebooks for mechanics, abilities, and game system information.

    Use this tool for questions about:
    - Game mechanics and rules
    - Character abilities and talents
    - Combat rules and actions
    - Ship systems and specifications
    - Equipment and items

    Args:
        ctx: Agent runtime context with state dependencies
        query: Search query text
        match_count: Number of results to return (default: 20)

    Returns:
        String containing the retrieved rules information
    """
    try:
        agent_deps = AgentDependencies()
        await agent_deps.initialize()

        class DepsWrapper:
            def __init__(self, deps):
                self.deps = deps

        deps_ctx = DepsWrapper(agent_deps)

        results = await hybrid_search(
            ctx=deps_ctx,  # type: ignore[arg-type]
            query=query,
            match_count=match_count,
            source_filter=RULES_FILTER
        )

        await agent_deps.cleanup()

        # Get state for citation map population
        state = ctx.deps.state if ctx.deps else None
        return await format_search_results(results, state)

    except Exception as e:
        return f"Error searching rules: {str(e)}"


@rag_agent.tool
async def search_game_logs(
    ctx: RunContext[StateDeps[RAGState]],
    query: str,
    match_count: Optional[int] = 20
) -> str:
    """
    Search the game session logs and campaign history.

    Use this tool for questions about:
    - Past events and story developments
    - Character actions and decisions
    - NPC interactions and relationships
    - Location visits and discoveries
    - Campaign timeline and history

    Args:
        ctx: Agent runtime context with state dependencies
        query: Search query text
        match_count: Number of results to return (default: 20)

    Returns:
        String containing the retrieved session information
    """
    try:
        agent_deps = AgentDependencies()
        await agent_deps.initialize()

        class DepsWrapper:
            def __init__(self, deps):
                self.deps = deps

        deps_ctx = DepsWrapper(agent_deps)

        results = await hybrid_search(
            ctx=deps_ctx,  # type: ignore[arg-type]
            query=query,
            match_count=match_count,
            source_filter=GAME_LOGS_FILTER
        )

        await agent_deps.cleanup()

        # Get state for citation map population
        state = ctx.deps.state if ctx.deps else None
        return await format_search_results(results, state)

    except Exception as e:
        return f"Error searching game logs: {str(e)}"
