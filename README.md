# infinite chat

A terminal chat client with conversation history management based on Retrieval-Augmented Language Models (RLM). Originally based on research by Alex L. Zhang and Omar Khattab from MIT OASYS lab.

This implementation has only been minimally tested with GLM 4.6. I wanted to code something for personal use and see if a simple implementation could work as a permanent to-do list.

## What It Does

- Terminal chat client with persistent conversation storage
- Intelligent search through conversation history using fuzzy matching
- Context window management with automatic sliding window (~200k tokens)
- LLM can search and expand conversation context when needed
- Secondary conversation for things you don't want to be remembered for (switch with /switch)
- Local file storage for all conversations
- There are also some slash commands that add semi-useful functionality, currently undocumented

## Setup

```bash
# Install dependencies
uv sync

# Configure API key
cp .env.example .env
# Edit .env with your provider details

# Run the launcher
./chat.sh
```

## Manual Setup

```bash
# Terminal 1: Start server
uv run python main.py

# Terminal 2: Start client
uv run python client.py
```

## API Endpoints

Server runs on port 8421:

- `POST /api/chat` - Send messages with history access
- `POST /api/search` - Search conversation history
- `POST /api/expand` - Expand context around messages
- `GET /api/history/{conversation_id}` - Get conversation history
- `GET /api/health` - Health check

## Architecture

- `storage.py` - JSON file operations
- `context.py` - Context window logic
- `search.py` - Fuzzy search implementation
- `llm.py` - LLM API wrapper
- `api.py` - FastAPI endpoints
- `main.py` - Server entry point
- `client.py` - Terminal client
- `chat.sh` - Launcher script

## Environment Configuration

Supported providers in `.env`:
- `ZAI_API_KEY`, `ZAI_BASE_URL` - Z.AI GLM-4.6 (default)
- `OPENAI_API_KEY`, `OPENAI_BASE_URL` - OpenAI models
- `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` - Claude models
- `OLLAMA_BASE_URL` - Ollama local models