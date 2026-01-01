"""Settings configuration for MongoDB RAG Agent."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # MongoDB Configuration
    mongodb_uri: str = Field(..., description="MongoDB Atlas connection string")

    mongodb_database: str = Field(default="rag_db", description="MongoDB database name")

    mongodb_collection_documents: str = Field(
        default="documents", description="Collection for source documents"
    )

    mongodb_collection_chunks: str = Field(
        default="chunks", description="Collection for document chunks with embeddings"
    )

    mongodb_vector_index: str = Field(
        default="vector_index",
        description="Vector search index name (must be created in Atlas UI)",
    )

    mongodb_text_index: str = Field(
        default="text_index",
        description="Full-text search index name (must be created in Atlas UI)",
    )

    # LLM Configuration (OpenAI-compatible)
    llm_provider: str = Field(
        default="openrouter",
        description="LLM provider (openai, anthropic, gemini, ollama, etc.)",
    )

    llm_api_key: str = Field(..., description="API key for the LLM provider")

    llm_model: str = Field(
        default="anthropic/claude-haiku-4.5",
        description="Model to use for search and summarization",
    )

    llm_base_url: Optional[str] = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the LLM API (for OpenAI-compatible providers)",
    )

    # Embedding Configuration
    embedding_provider: str = Field(default="openai", description="Embedding provider")

    embedding_api_key: str = Field(..., description="API key for embedding provider")

    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model to use"
    )

    embedding_base_url: Optional[str] = Field(
        default="https://api.openai.com/v1", description="Base URL for embedding API"
    )

    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension (1536 for text-embedding-3-small)",
    )

    # Search Configuration
    default_match_count: int = Field(
        default=10, description="Default number of search results to return"
    )

    max_match_count: int = Field(
        default=50, description="Maximum number of search results allowed"
    )

    default_text_weight: float = Field(
        default=0.3, description="Default text weight for hybrid search (0-1)"
    )

    # Slack Bot Configuration (Socket Mode)
    slack_bot_token: str = Field(
        default="",
        description="Slack Bot OAuth Token (xoxb-...)"
    )

    slack_app_token: str = Field(
        default="",
        description="Slack App Token for Socket Mode (xapp-...)"
    )

    slack_signing_secret: str = Field(
        default="",
        description="Slack signing secret for request verification"
    )

    mongodb_collection_conversations: str = Field(
        default="conversations",
        description="MongoDB collection for Slack conversation history"
    )

    # Komga Configuration (PDF hosting with deep links)
    komga_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for Komga server (e.g., https://komga.example.com)"
    )

    komga_username: Optional[str] = Field(
        default=None,
        description="Komga username for API authentication"
    )

    komga_password: Optional[str] = Field(
        default=None,
        description="Komga password for API authentication"
    )

    komga_cache_file: str = Field(
        default=".komga_cache.json",
        description="Path to local JSON file for caching book ID lookups"
    )

    # Slack Bot UX
    slack_thinking_messages: list[str] = Field(
        default=[
            "_Thinking..._",
            "_Processing..._",
            "_Searching the knowledge base..._",
            "_Let me check..._",
            "_Consulting my superior intellect..._",
            "_*sigh* Fine, looking that up..._",
            "_Running calculations..._",
        ],
        description="Random thinking indicator messages shown while processing"
    )


def load_settings() -> Settings:
    """Load settings with proper error handling."""
    try:
        return Settings()  # type: ignore[call-arg]
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        if "mongodb_uri" in str(e).lower():
            error_msg += "\nMake sure to set MONGODB_URI in your .env file"
        if "llm_api_key" in str(e).lower():
            error_msg += "\nMake sure to set LLM_API_KEY in your .env file"
        if "embedding_api_key" in str(e).lower():
            error_msg += "\nMake sure to set EMBEDDING_API_KEY in your .env file"
        raise ValueError(error_msg) from e
