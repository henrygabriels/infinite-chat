# infinite chat

A terminal chat client with tool-based conversation history access, based (loosely - no secondary model) on Retrieval-Augmented Language Models (RLM). Originally based on [research by Alex L. Zhang and Omar Khattab from MIT OASYS lab](https://alexzhang13.github.io/blog/2025/rlm/).

It will *ideally* give me (and maybe you!) one chat context + retrieval forever, in the terminal, for all the errata / to-do stuff which God knows where I saved. I plan on just leaving this open in one of my 'always open' terminal windows for a few weeks to see what happens.


This implementation has only been minimally tested with GLM 4.6 in Ghostty on Mac "OS 26". The code is clean, but the actual action of doing bad RAG (brag?) - a fuzzy search with more features - on my own conversation history (where I probably repeat the same words often) is untested, since the paper is so new, and accumulating real history takes time.

## What It Does

- Extremely minimally-styled terminal chat client with persistent conversation storage in JSON
- Tool access semi-intelligent search through conversation history using fuzzy matching
- Context window 'management' with automatic poor-man's sliding window (~200k tokens)
- LLM can search and expand conversation context when needed (theoretically!)
- A 'secondary conversation' available via /switch for whatever you don't want to be remembered for - goes to an entirely separate JSON file, loads an entirely separate context.
- Local file storage for all conversations
- There are also some slash commands that add semi-useful functionality, currently undocumented.
- In future, if it's useful, I might build some different frontends for remote access if SSHing in is too annoying.

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
