# Anne Bonny - Expanse RPG Knowledge Agent

Agentic RAG system for The Expanse tabletop RPG, featuring Anne Bonny - the sardonic AI managing a pirate spaceship. Built with MongoDB Atlas Vector Search, Pydantic AI, and hybrid search that works on the free tier.

## Features

- **Hybrid Search**: Semantic + keyword search with Reciprocal Rank Fusion (works on free M0 tier)
- **Multi-Format Ingestion**: PDF, Word, PowerPoint, Excel, HTML, Markdown, audio transcription
- **Intelligent Chunking**: Docling HybridChunker preserves document structure and page numbers
- **Dual Interfaces**: Rich CLI with streaming + Slack bot with Socket Mode
- **Citation Deep Links**: Komga integration for clickable PDF page references
- **Incremental Updates**: SHA256-based change detection for efficient re-ingestion
- **Specialized Tools**: Separate search for rules (GRR PDFs) vs. game session logs

## Prerequisites

- Python 3.10+
- MongoDB Atlas account (free M0 tier works)
- LLM API key (OpenRouter, OpenAI, or compatible)
- Embedding API key (OpenAI recommended)
- [UV package manager](https://docs.astral.sh/uv/)

**Optional:**
- Slack app tokens (for Slack bot)
- Komga instance (for PDF deep linking)

## Quick Start

### 1. Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup

```bash
git clone https://github.com/jflo/MongoDB-RAG-Agent.git
cd MongoDB-RAG-Agent

uv venv
source .venv/bin/activate  # Unix/Mac
# .venv\Scripts\activate   # Windows
uv sync
```

### 3. Set Up MongoDB Atlas

1. Create a free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
2. Create an M0 Free cluster
3. Configure security:
   - Create database user (save credentials)
   - Add your IP to Network Access
4. Get connection string: Connect > Drivers > Copy URI
   - Replace `<password>` with your actual password

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# MongoDB
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGODB_DATABASE=rag_db

# LLM (OpenRouter example)
LLM_PROVIDER=openrouter
LLM_API_KEY=sk-or-v1-...
LLM_MODEL=anthropic/claude-sonnet-4

# Embeddings
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
```

### 5. Validate Configuration

```bash
uv run python -m src.test_config
```

### 6. Ingest Documents

```bash
# Add your PDFs to documents/ folder, then:
uv run python -m src.ingestion.ingest -d ./documents

# Incremental mode (skip unchanged files):
uv run python -m src.ingestion.ingest -d ./documents -i
```

### 7. Create Search Indexes

In MongoDB Atlas: Database > Search and Vector Search > Create Search Index

**Vector Search Index** (name: `vector_index`):
```json
{
  "fields": [{
    "type": "vector",
    "path": "embedding",
    "numDimensions": 1536,
    "similarity": "cosine"
  }]
}
```

**Atlas Search Index** (name: `text_index`):
```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "content": {
        "type": "string",
        "analyzer": "lucene.standard"
      }
    }
  }
}
```

Wait for indexes to show "Active" status.

### 8. Run the Agent

```bash
# CLI interface
uv run python -m src.cli

# Or Slack bot (see Slack Setup below)
uv run python -m src.slack_bot
```

## Slack Bot Setup

### 1. Create Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) > Create New App > From scratch
2. Enable **Socket Mode** (Settings > Socket Mode > Enable)
3. Generate an **App-Level Token** with `connections:write` scope

### 2. Configure Bot Permissions

OAuth & Permissions > Bot Token Scopes:
- `app_mentions:read` - Receive @mentions
- `chat:write` - Send messages
- `channels:history` - Read channel messages (for context)

### 3. Enable Events

Event Subscriptions > Enable Events > Subscribe to bot events:
- `app_mention` - Trigger on @BotName

### 4. Install to Workspace

OAuth & Permissions > Install to Workspace > Copy Bot Token

### 5. Configure Environment

Add to `.env`:
```bash
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

### 6. Run

```bash
uv run python -m src.slack_bot
```

Mention the bot in any channel it's invited to: `@AnneBonny what are the grappling rules?`

## Komga Integration

Komga integration enables clickable citations that link directly to PDF pages in your Komga reader.

### Configuration

Add to `.env`:
```bash
KOMGA_BASE_URL=https://komga.example.com
KOMGA_USERNAME=your_username
KOMGA_PASSWORD=your_password
```

### How It Works

1. During search, page numbers are extracted from chunk metadata
2. Komga API maps filenames to book IDs (cached locally in `.komga_cache.json`)
3. Citations like `(GRR6610_TheExpanse_TUE_Core.pdf, pp. 42-45)` become clickable links
4. Links open directly to the referenced page in Komga's reader

### Cache Management

The filename-to-bookId mapping is cached to avoid repeated API calls. Delete `.komga_cache.json` to refresh.

## Project Structure

```
src/
├── agent.py              # Pydantic AI agent with search tools
├── agent_runner.py       # Shared agent execution logic
├── cli.py                # Rich-based CLI with streaming
├── slack_bot.py          # Slack Socket Mode interface
├── tools.py              # Search tools (semantic, text, hybrid)
├── prompts.py            # Anne Bonny system prompt
├── settings.py           # Pydantic Settings configuration
├── providers.py          # LLM/embedding provider setup
├── dependencies.py       # MongoDB client injection
├── komga.py              # Komga PDF deep linking
├── response_filter.py    # Clean LLM output for display
├── conversation_store.py # Slack history storage
├── test_config.py        # Configuration validator
└── ingestion/
    ├── ingest.py         # Document ingestion pipeline
    ├── chunker.py        # Docling HybridChunker wrapper
    └── embedder.py       # Batch embedding generation

documents/                # Your source documents
.komga_cache.json         # Komga book ID cache (generated)
```

## Ingestion Options

```bash
uv run python -m src.ingestion.ingest [options]

Options:
  -d, --documents PATH    Documents folder (default: documents)
  -i, --incremental       Skip unchanged files, update changed
  --no-clean              Don't clear existing data first
  --chunk-size N          Target chunk size (default: 1000)
  --chunk-overlap N       Chunk overlap (default: 200)
  --max-tokens N          Max tokens per chunk (default: 512)
  -v, --verbose           Debug logging
```

### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | .pdf | Full text + structure extraction |
| Word | .docx, .doc | |
| PowerPoint | .pptx, .ppt | |
| Excel | .xlsx, .xls | |
| HTML | .html, .htm | |
| Markdown | .md | YAML frontmatter preserved |
| Text | .txt | |
| Audio | .mp3, .wav, .m4a, .flac | Whisper transcription |

## Search Tools

The agent has three specialized search tools:

### `search_knowledge_base`
General search across all documents. Supports `semantic`, `text`, or `hybrid` modes.

### `search_rules`
Filtered to `GRR*.pdf` - Green Ronin rulebooks only. Use for game mechanics, abilities, combat rules.

### `search_game_logs`
Filtered to `GMT*.transcript_summary.md` - Session transcripts. Use for campaign history, past events.

### Customizing Filters

Edit `src/agent.py` to modify source patterns:
```python
RULES_FILTER = r"^GRR.*\.pdf$"
GAME_LOGS_FILTER = r"^GMT.*\.transcript_summary\.md$"
```

## Development

### Install Dev Dependencies

```bash
uv sync --group dev
```

### Run Tests

```bash
uv run pytest tests/ -v
```

### Code Style

```bash
# Format
uv run black src/ tests/

# Lint
uv run ruff check src/ tests/
```

### Type Safety

All code must have type annotations. See `CLAUDE.md` for full development guidelines.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Database | MongoDB Atlas (Vector + Atlas Search) |
| Agent Framework | Pydantic AI 0.1+ |
| Document Processing | Docling 2.14+ |
| Audio Transcription | OpenAI Whisper |
| CLI | Rich 13.9+ |
| Slack | slack-bolt with Socket Mode |
| HTTP Client | httpx (async) |
| Package Manager | UV |

## How Hybrid Search Works

This project implements Reciprocal Rank Fusion (RRF) manually, enabling hybrid search on MongoDB's free M0 tier (the `$rankFusion` operator requires paid tiers).

1. **Semantic Search** (`$vectorSearch`): Vector similarity on embeddings
2. **Text Search** (`$search`): Keyword matching with fuzzy support
3. **RRF Merge**: `score = 1/(k + rank)` where k=60, combined across both result sets

Both searches run concurrently for ~350-600ms latency.

## License

MIT
