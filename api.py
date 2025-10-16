from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import uuid
import json

from storage import ConversationStorage
from context import ContextWindow
from search import FuzzySearch
from llm import LLMClient

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    context_window_size: int = 200000

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    context_stats: Dict[str, int]

class SearchRequest(BaseModel):
    conversation_id: str
    query: str
    limit: int = 5

class ExpandRequest(BaseModel):
    conversation_id: str
    message_id: str
    direction: str = "both"
    pairs: int = 3

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]

class ExpandResponse(BaseModel):
    messages: List[Dict[str, Any]]

class HistoryResponse(BaseModel):
    messages: List[Dict[str, Any]]
    total_count: int

# Initialize components
app = FastAPI(title="infinite chat", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
storage = ConversationStorage()
context_window = ContextWindow()
search = FuzzySearch()
llm_client = None  # Lazy initialization

def get_llm_client():
    """Lazy initialization of LLM client."""
    global llm_client
    if llm_client is None:
        llm_client = LLMClient()
    return llm_client

async def execute_tool_calls(tool_calls: List[Dict[str, Any]], conversation_id: str) -> List[Dict[str, Any]]:
    """Execute tool calls from the LLM."""
    tool_results = []

    for tool_call in tool_calls:
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])

        try:
            if function_name == "search_conversations":
                messages = storage.load_conversation(conversation_id)
                results = search.search_messages(messages, arguments["query"], arguments.get("limit", 5))
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "content": json.dumps(results)
                })

            elif function_name == "expand_context":
                messages = storage.load_conversation(conversation_id)
                expanded = search.expand_context(
                    messages,
                    arguments["message_id"],
                    arguments.get("direction", "both"),
                    arguments.get("pairs", 3)
                )
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "content": json.dumps(expanded)
                })

        except Exception as e:
            tool_results.append({
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "content": json.dumps({"error": str(e)})
            })

    return tool_results

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get response with tool support."""
    try:
        # Use provided conversation_id or create new one
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Load conversation
        messages = storage.load_conversation(conversation_id)

        # Add user message
        user_message_id = storage.append_message(conversation_id, "user", request.message)
        messages.append({
            "role": "user",
            "content": request.message,
            "id": user_message_id
        })

        # Get context window
        context_messages = context_window.get_context_window(messages, reserve_tokens=20000)

        # Initial chat request
        client = get_llm_client()
        response = await client.chat(context_messages, {}, request.context_window_size)

        # Handle tool calls if present
        if "tool_calls" in response.get("choices", [{}])[0].get("message", {}):
            assistant_message = response["choices"][0]["message"]
            tool_calls = assistant_message["tool_calls"]

            # Add assistant message with tool calls
            context_messages.append(assistant_message)

            # Execute tools
            tool_results = await execute_tool_calls(tool_calls, conversation_id)
            context_messages.extend(tool_results)

            # Get final response after tool execution
            final_response = await client.chat(context_messages, {}, request.context_window_size)

            assistant_content = final_response["choices"][0]["message"]["content"]
        else:
            assistant_content = response["choices"][0]["message"]["content"]

        # Save assistant response
        assistant_message_id = storage.append_message(conversation_id, "assistant", assistant_content)

        # Get context stats
        updated_messages = storage.load_conversation(conversation_id)
        stats = context_window.get_window_stats(context_window.get_context_window(updated_messages))

        return ChatResponse(
            response=assistant_content,
            conversation_id=conversation_id,
            message_id=assistant_message_id,
            context_stats=stats
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search", response_model=SearchResponse)
async def search_conversations(request: SearchRequest):
    """Search conversation history."""
    try:
        messages = storage.load_conversation(request.conversation_id)
        results = search.search_messages(messages, request.query, request.limit)
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/expand", response_model=ExpandResponse)
async def expand_context(request: ExpandRequest):
    """Expand context around a message."""
    try:
        messages = storage.load_conversation(request.conversation_id)
        expanded = search.expand_context(
            messages,
            request.message_id,
            request.direction,
            request.pairs
        )
        return ExpandResponse(messages=expanded)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{conversation_id}", response_model=HistoryResponse)
async def get_history(conversation_id: str, limit: int = 50, offset: int = 0):
    """Get conversation history with pagination."""
    try:
        messages = storage.load_conversation(conversation_id)
        total_count = len(messages)

        # Apply pagination
        paginated_messages = messages[offset:offset + limit]

        return HistoryResponse(
            messages=paginated_messages,
            total_count=total_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations")
async def list_conversations():
    """List all conversations."""
    try:
        conversations = storage.list_conversations()
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}