#!/usr/bin/env python3
"""
LLM Chat Client MVP
Minimal, no-bloat backend with intelligent conversation history management.
"""

import uvicorn
from api import app

if __name__ == "__main__":
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("                                         ")
    print("      infinite chat launcher              ")
    print("                                         ")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("")
    print("API Documentation: http://localhost:8421/docs")
    print("Health Check: http://localhost:8421/api/health")
    print("")
    print("Environment variables needed:")
    print("  Copy .env.example to .env and configure your LLM provider")
    print("")
    print("Supported providers:")
    print("  - Z.AI (default): ZAI_API_KEY, ZAI_BASE_URL")
    print("  - OpenAI: OPENAI_API_KEY, OPENAI_BASE_URL")
    print("  - Anthropic: ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL")
    print("  - Ollama: OLLAMA_BASE_URL (local, no API key needed)")
    print("")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8421,
        reload=True,
        log_level="info",
        timeout_keep_alive=1200  # 20 minutes keep-alive for long RLM processing
    )