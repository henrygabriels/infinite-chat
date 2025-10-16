import os
import json
from typing import List, Dict, Any, Optional
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "zai")
        self.client = httpx.AsyncClient(timeout=120.0)

        if self.provider == "zai":
            self.api_key = os.getenv("ZAI_API_KEY")
            self.base_url = os.getenv("ZAI_BASE_URL", "https://api.z.ai/v1")
            self.model = os.getenv("ZAI_MODEL", "glm-4.6")
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        elif self.provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        elif self.provider == "ollama":
            self.api_key = "ollama"  # Ollama doesn't use API keys
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            self.model = os.getenv("OLLAMA_MODEL", "llama3.1")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        if not self.api_key and self.provider != "ollama":
            raise ValueError(f"API key not found for provider: {self.provider}")

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Define the tools available to the LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_conversations",
                    "description": "Search conversation history using fuzzy matching. Returns snippets around matching text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query - uses fuzzy matching like fzf"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "expand_context",
                    "description": "Expand context around a specific message. Use this when you need to see full message pairs around a search result.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message_id": {
                                "type": "string",
                                "description": "ID of the message to expand around"
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["before", "after", "both"],
                                "description": "Which direction to expand around the message",
                                "default": "both"
                            },
                            "pairs": {
                                "type": "integer",
                                "description": "Number of message pairs to expand (each pair = 2 messages)",
                                "default": 3
                            }
                        },
                        "required": ["message_id"]
                    }
                }
            }
        ]

    def get_system_prompt(self, context_window_size: int) -> str:
        """Generate system prompt with tool instructions."""
        return f"""You are an AI assistant with access to conversation history. You can see the most recent messages in context, but also have tools to search and expand older conversations when needed.

**Context Window**: You currently see the ~{context_window_size // 1000}k most recent tokens of conversation.

**Available Tools**:
1. **search_conversations(query, limit)**: Search through your entire conversation history using fuzzy matching. Returns relevant snippets with message IDs.
2. **expand_context(message_id, direction, pairs)**: Get full message pairs around a specific message ID. Use this to "scroll" through conversations.

**When to Search**:
- User references previous discussions
- You need context beyond the current window
- User asks "what did we discuss about..." or similar
- Following up on earlier topics
- You feel like you're missing important context

**Search Strategy**:
1. Start with broad searches using relevant keywords
2. Review the snippets to identify promising matches
3. Use expand_context on the most relevant message IDs
4. "Scroll" by expanding more pairs if needed
5. Use the recovered context to answer the original question

**Example Usage**:
User: "What did we decide about the database?"
Your process:
1. search_conversations("database", limit=5)
2. Review snippets, find relevant message IDs
3. expand_context("msg_123", "both", 3)
4. Answer using the recovered context

Remember: The conversation history is infinite, but you can intelligently navigate it using these tools."""

    async def chat(self, messages: List[Dict[str, Any]], tools: Dict[str, Any],
                   context_window_size: int = 200000) -> Dict[str, Any]:
        """Send chat request to LLM API with tools."""

        # Prepare the messages with system prompt
        system_message = {
            "role": "system",
            "content": self.get_system_prompt(context_window_size)
        }

        request_messages = [system_message] + messages

        # Prepare the request
        payload = {
            "model": self.model,
            "messages": request_messages,
            "tools": self.get_tools_schema(),
            "tool_choice": "auto",
            "stream": False,
            "max_tokens": 4000,
            "temperature": 0.7
        }

        # Set headers based on provider
        headers = {"Content-Type": "application/json"}

        if self.provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        elif self.provider == "ollama":
            # Ollama doesn't use authentication headers
            pass
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            raise Exception(f"{self.provider.upper()} API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise Exception(f"Error calling {self.provider.upper()} API: {str(e)}")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any],
                         messages: List[Dict[str, Any]]) -> Any:
        """Execute tool calls - this will be implemented by the API layer."""
        # This method is a placeholder - actual tool execution happens in api.py
        # This allows for proper dependency injection and testing
        pass