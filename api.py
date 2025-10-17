from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import uuid
import json

from storage import ConversationStorage
from context import ContextWindow
from search import FuzzySearch
from llm import LLMClient
from rlm_agent import RLMAgent
from true_rlm_agent import TrueRLMAgent
from rlm_storage import RLMStorage

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

class RLMChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    context_window_size: int = 200000

class RLMChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    context_stats: Dict[str, int]
    rlm_stats: Dict[str, Any]

class RLMLogsResponse(BaseModel):
    agent_logs: List[Dict[str, Any]]
    conversation_logs: List[Dict[str, Any]]
    stats: Dict[str, Any]

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

# Add GZip middleware for compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global instances
storage = ConversationStorage()
context_window = ContextWindow()
search = FuzzySearch()
llm_client = None  # Lazy initialization
rlm_storage = RLMStorage()
rlm_agent = None  # Lazy initialization
true_rlm_agent = None  # Lazy initialization

def get_llm_client():
    """Lazy initialization of LLM client."""
    global llm_client
    if llm_client is None:
        llm_client = LLMClient()
    return llm_client

def get_rlm_agent():
    """Lazy initialization of RLM agent."""
    global rlm_agent
    if rlm_agent is None:
        client = get_llm_client()
        rlm_agent = RLMAgent(client, rlm_storage, search)
    return rlm_agent

def get_true_rlm_agent():
    """Lazy initialization of True RLM agent."""
    global true_rlm_agent
    if true_rlm_agent is None:
        client = get_llm_client()
        true_rlm_agent = TrueRLMAgent(client, rlm_storage, search)
    return true_rlm_agent

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

@app.post("/api/rlm-chat", response_model=RLMChatResponse)
async def rlm_chat(request: RLMChatRequest):
    """True RLM mode chat with strategic context access by Root LM."""
    try:
        # Use provided conversation_id or create new one
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Initialize RLM mode if this is a new conversation
        if not rlm_storage.is_rlm_conversation(conversation_id):
            rlm_storage.switch_to_rlm_mode(conversation_id)

        # Step 1: Save user message to clean RLM conversation
        user_message_id = rlm_storage.append_rlm_message(conversation_id, "user", request.message)

        # Step 2: Process through True RLM agent (Root LM with strategic context access)
        agent = get_true_rlm_agent()
        rlm_result = await agent.process_user_query(conversation_id, request.message)

        # Step 3: Save the complete RLM processing log to agent log
        agent_metadata = {
            "original_query": request.message,
            "rlm_pattern": "true_rlm",
            "iterations": rlm_result.get("iterations", 0),
            "answer_length": len(rlm_result.get("answer", "")),
            "has_reasoning": bool(rlm_result.get("reasoning")),
            "context_sources_count": len(rlm_result.get("context_sources", []))
        }

        rlm_storage.append_rlm_agent_message(
            conversation_id,
            "system",
            f"True RLM processing for query: {request.message}",
            agent_metadata
        )

        # Save the full conversation log
        if "conversation_log" in rlm_result:
            for log_entry in rlm_result["conversation_log"]:
                rlm_storage.append_rlm_agent_message(
                    conversation_id,
                    log_entry.get("role", "system"),
                    log_entry.get("content", ""),
                    {
                        "timestamp": log_entry.get("timestamp"),
                        "type": log_entry.get("type", "log_entry")
                    }
                )

        # Step 4: Extract the final answer
        final_answer = rlm_result.get("answer", "I apologize, but I couldn't process your request.")
        reasoning = rlm_result.get("reasoning", "")

        # Step 5: Save clean assistant response to RLM conversation
        assistant_message_id = rlm_storage.append_rlm_message(conversation_id, "assistant", final_answer)

        # Step 6: Get context and RLM stats
        updated_messages = rlm_storage.load_rlm_conversation(conversation_id)
        context_stats = context_window.get_window_stats(context_window.get_context_window(updated_messages))
        rlm_stats = rlm_storage.get_rlm_stats(conversation_id)

        # Add True RLM specific stats
        rlm_stats.update({
            "rlm_pattern": "true_rlm",
            "iterations_used": rlm_result.get("iterations", 0),
            "has_reasoning": bool(reasoning),
            "context_sources": rlm_result.get("context_sources", []),
            "processing_time_seconds": rlm_result.get("processing_time_seconds", 0)
        })

        return RLMChatResponse(
            response=final_answer,
            conversation_id=conversation_id,
            message_id=assistant_message_id,
            context_stats=context_stats,
            rlm_stats=rlm_stats
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rlm-logs/{conversation_id}", response_model=RLMLogsResponse)
async def get_rlm_logs(conversation_id: str):
    """Get RLM agent logs and conversation logs."""
    try:
        # Get agent logs
        agent_logs = rlm_storage.load_rlm_agent_conversation(conversation_id)

        # Get conversation logs
        conversation_logs = rlm_storage.load_rlm_conversation(conversation_id)

        # Get stats
        stats = rlm_storage.get_rlm_stats(conversation_id)

        return RLMLogsResponse(
            agent_logs=agent_logs,
            conversation_logs=conversation_logs,
            stats=stats
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}