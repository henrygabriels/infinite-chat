#!/usr/bin/env python3
"""
True RLM Agent - Root LM with Programmatic Context Access
Implements the RLM pattern from the paper: LM with strategic context environment access.
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from llm import LLMClient
from rlm_storage import RLMStorage
from search import FuzzySearch
from datetime import datetime


class TrueRLMAgent:
    """
    True RLM Agent that gives the LM programmatic control over context exploration.
    Root LM receives only the user's query and uses tools to strategically access context.
    """

    def __init__(self, llm_client: LLMClient, rlm_storage: RLMStorage, search: FuzzySearch):
        self.llm_client = llm_client
        self.rlm_storage = rlm_storage
        self.search = search

    def get_rlm_tools_schema(self) -> List[Dict[str, Any]]:
        """Define tools for strategic context access following the RLM pattern."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_context_overview",
                    "description": "Get metadata and overview of available conversation context",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_context_chunk",
                    "description": "Retrieve a specific chunk of conversation context by index range",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_index": {
                                "type": "integer",
                                "description": "Starting index of messages to retrieve"
                            },
                            "end_index": {
                                "type": "integer",
                                "description": "Ending index (exclusive), defaults to start_index + 10"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to return, defaults to 2000"
                            }
                        },
                        "required": ["start_index"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_context",
                    "description": "Search through conversation context to find relevant sections",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for finding relevant context"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return, defaults to 5"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "recursive_lm_call",
                    "description": "Call LM recursively on a specific context subset for analysis or processing",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Prompt for the recursive LM call"
                            },
                            "context_subset": {
                                "type": "array",
                                "description": "Array of message objects to analyze"
                            },
                            "task": {
                                "type": "string",
                                "enum": ["summarize", "analyze", "extract", "compare", "synthesize"],
                                "description": "Type of analysis task for the recursive LM"
                            }
                        },
                        "required": ["prompt", "context_subset"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "description": "Provide the final response to the user's original query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "Final answer to the user's question"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of how you arrived at this answer (optional)"
                            },
                            "context_sources": {
                                "type": "array",
                                "description": "List of context sources used in the answer (optional)"
                            }
                        },
                        "required": ["answer"]
                    }
                }
            }
        ]

    def get_root_lm_system_prompt(self) -> str:
        """System prompt for the root LM following the RLM pattern."""
        return """You are a Root LM in a Retrieval-Augmented Language Model (RLM) system.

**Your Role**: You receive ONLY the user's query and have access to tools for strategically exploring conversation context. You must decide how to programmatically access and analyze the context to answer the user's question.

**Available Context Environment**:
- Full conversation history is stored as accessible data
- You can retrieve chunks, search, and make recursive LM calls
- Context contains message pairs (user: "question", assistant: "response")

**Your Strategic Approach**:
1. Start with context overview to understand what's available
2. Search for relevant sections based on the user's query
3. Retrieve specific chunks for detailed analysis
4. Use recursive LM calls to analyze/synthesize context subsets
5. Provide final answer with reasoning

**Important Guidelines**:
- Start broad, then drill down systematically
- Use recursive_lm_call for complex analysis tasks
- Always provide final_answer when you're ready to respond
- Be strategic about context access - don't retrieve everything at once
- Use your tools efficiently to find the most relevant information

**Tool Usage Strategy**:
- get_context_overview() → Understand available conversation scope
- search_context(query) → Find relevant conversation sections
- get_context_chunk(start, end) → Retrieve specific message ranges
- recursive_lm_call() → Analyze or synthesize context subsets
- final_answer() → Provide your response when ready

The user's question is: "{user_query}"

Begin your strategic context exploration now."""

    def get_recursive_lm_system_prompt(self) -> str:
        """System prompt for recursive LM calls."""
        return """You are a Recursive LM called by the Root LM for specific analysis tasks.

**Your Role**: Analyze the provided context subset according to the given task and prompt. Focus on extracting insights, patterns, or information that will help answer the original user question.

**Available Tasks**:
- summarize: Create concise summaries of context
- analyze: Deep analysis of content and patterns
- extract: Pull out specific information or data
- compare: Compare different parts of context
- synthesize: Combine multiple pieces of information

**Guidelines**:
- Stay focused on your specific task
- Be thorough but concise
- Provide actionable insights
- Reference the context directly when helpful

Return your analysis as a clear, structured response."""

    async def execute_context_tool(self, tool_name: str, arguments: Dict[str, Any],
                                 conversation_id: str) -> Dict[str, Any]:
        """Execute context access tools for the Root LM."""

        if tool_name == "get_context_overview":
            return await self._get_context_overview(conversation_id)

        elif tool_name == "get_context_chunk":
            return await self._get_context_chunk(conversation_id, arguments)

        elif tool_name == "search_context":
            return await self._search_context(conversation_id, arguments)

        elif tool_name == "recursive_lm_call":
            return await self._recursive_lm_call(conversation_id, arguments)

        elif tool_name == "final_answer":
            return {"type": "final_answer", "data": arguments}

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def _get_context_overview(self, conversation_id: str) -> Dict[str, Any]:
        """Get overview metadata about available context."""
        try:
            full_context = self.rlm_storage.get_full_history_for_search(conversation_id)

            if not full_context:
                return {
                    "total_messages": 0,
                    "total_tokens": 0,
                    "conversation_span": "No messages",
                    "topics": [],
                    "message_distribution": {}
                }

            # Calculate overview statistics
            total_messages = len(full_context)
            total_tokens = sum(len(msg.get('content', '')) for msg in full_context)

            # Get time span
            timestamps = [msg.get('timestamp', '') for msg in full_context if msg.get('timestamp')]
            if timestamps:
                time_span = f"{timestamps[0][:10]} to {timestamps[-1][:10]}"
            else:
                time_span = "Unknown time span"

            # Simple topic analysis (extract key terms)
            all_text = ' '.join([msg.get('content', '').lower() for msg in full_context])
            common_words = [word for word in all_text.split()
                          if len(word) > 4 and all_text.count(word) > 2][:10]

            # Message distribution by role
            role_counts = {}
            for msg in full_context:
                role = msg.get('role', 'unknown')
                role_counts[role] = role_counts.get(role, 0) + 1

            return {
                "total_messages": total_messages,
                "total_tokens": total_tokens,
                "conversation_span": time_span,
                "potential_topics": common_words,
                "message_distribution": role_counts,
                "context_available": True
            }

        except Exception as e:
            return {"error": f"Failed to get context overview: {str(e)}"}

    async def _get_context_chunk(self, conversation_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve a specific chunk of context by index range."""
        try:
            full_context = self.rlm_storage.get_full_history_for_search(conversation_id)

            start_index = args.get('start_index', 0)
            end_index = args.get('end_index', start_index + 10)
            max_tokens = args.get('max_tokens', 2000)

            # Validate indices
            if start_index < 0:
                start_index = 0
            if end_index > len(full_context):
                end_index = len(full_context)
            if start_index >= len(full_context):
                return {"error": f"Start index {start_index} exceeds total messages {len(full_context)}"}

            # Extract chunk
            chunk = full_context[start_index:end_index]

            # Apply token limit if needed
            current_tokens = sum(len(msg.get('content', '')) for msg in chunk)
            if current_tokens > max_tokens:
                # Reduce chunk size by removing messages from the end
                temp_chunk = []
                running_tokens = 0
                for msg in chunk:
                    msg_tokens = len(msg.get('content', ''))
                    if running_tokens + msg_tokens <= max_tokens:
                        temp_chunk.append(msg)
                        running_tokens += msg_tokens
                    else:
                        break
                chunk = temp_chunk

            return {
                "chunk": chunk,
                "start_index": start_index,
                "end_index": start_index + len(chunk),
                "total_in_chunk": len(chunk),
                "estimated_tokens": sum(len(msg.get('content', '')) for msg in chunk),
                "has_more": end_index < len(full_context)
            }

        except Exception as e:
            return {"error": f"Failed to get context chunk: {str(e)}"}

    async def _search_context(self, conversation_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search through context to find relevant sections."""
        try:
            full_context = self.rlm_storage.get_full_history_for_search(conversation_id)
            query = args.get('query', '')
            limit = args.get('limit', 5)

            if not full_context:
                return {"results": [], "total_messages": 0}

            # Use fuzzy search to find relevant messages
            search_results = self.search.search_messages(full_context, query, limit)

            # Expand around each search result to get context
            expanded_results = []
            for result in search_results:
                message_id = result.get('message_id')
                if message_id:
                    expanded = self.search.expand_context(
                        full_context, message_id, "both", pairs=2
                    )
                    expanded_results.append({
                        "search_result": result,
                        "expanded_context": expanded
                    })

            return {
                "results": expanded_results,
                "query": query,
                "total_found": len(expanded_results)
            }

        except Exception as e:
            return {"error": f"Failed to search context: {str(e)}"}

    async def _recursive_lm_call(self, conversation_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recursive LM call on specific context subset."""
        try:
            prompt = args.get('prompt', '')
            context_subset = args.get('context_subset', [])
            task = args.get('task', 'analyze')

            if not context_subset:
                return {"error": "No context subset provided for recursive LM call"}

            # Prepare messages for recursive LM
            recursive_messages = [
                {
                    "role": "system",
                    "content": self.get_recursive_lm_system_prompt()
                },
                {
                    "role": "user",
                    "content": f"Task: {task}\n\nPrompt: {prompt}\n\nContext to analyze:\n{json.dumps(context_subset, indent=2)}"
                }
            ]

            # Call LM
            response = await self.llm_client.chat(recursive_messages, {}, 200000, self.get_recursive_lm_system_prompt())

            if "choices" in response and response["choices"]:
                result = response["choices"][0]["message"]["content"]
            else:
                result = "Recursive LM call failed"

            return {
                "result": result,
                "task": task,
                "context_size": len(context_subset),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": f"Failed recursive LM call: {str(e)}"}

    async def process_user_query(self, conversation_id: str, user_query: str) -> Dict[str, Any]:
        """
        Process user query using true RLM pattern.
        Root LM gets only the query and decides how to explore context.
        """
        start_time = time.time()
        try:
            # Prepare Root LM conversation with only the user's query
            system_message = {
                "role": "system",
                "content": self.get_root_lm_system_prompt().format(user_query=user_query)
            }

            user_message = {
                "role": "user",
                "content": user_query
            }

            # Initial Root LM call
            response = await self.llm_client.chat(
                [user_message],  # Don't include system_message in the list, use custom prompt
                {"tools": self.get_rlm_tools_schema()},
                200000,
                system_message["content"]  # Use custom system prompt
            )

            # Process tool calls iteratively
            conversation_log = [
                {"role": "system", "content": "RLM Root LM started", "timestamp": datetime.now().isoformat()},
                {"role": "user", "content": user_query, "timestamp": datetime.now().isoformat()}
            ]

            current_messages = [system_message, user_message]
            max_iterations = 20  # Increased for deeper context exploration
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                # Check if LM made tool calls
                if "choices" in response and response["choices"]:
                    message = response["choices"][0]["message"]

                    if "tool_calls" in message:
                        # Log the LM's tool call request
                        conversation_log.append({
                            "role": "assistant",
                            "content": f"Tool calls: {json.dumps(message['tool_calls'], indent=2)}",
                            "timestamp": datetime.now().isoformat(),
                            "type": "tool_request"
                        })

                        # Execute tool calls
                        tool_results = []
                        final_answer_found = False

                        for tool_call in message["tool_calls"]:
                            tool_name = tool_call["function"]["name"]
                            arguments = json.loads(tool_call["function"]["arguments"])

                            # Execute the tool
                            result = await self.execute_context_tool(tool_name, arguments, conversation_id)

                            # Log tool execution
                            conversation_log.append({
                                "role": "tool",
                                "content": f"Tool '{tool_name}' result: {json.dumps(result, indent=2)}",
                                "timestamp": datetime.now().isoformat(),
                                "type": "tool_result"
                            })

                            # Check if this is a final answer
                            if result.get("type") == "final_answer":
                                final_answer_found = result
                                break

                            # Format result for next LM call
                            tool_results.append({
                                "tool_call_id": tool_call["id"],
                                "role": "tool",
                                "content": json.dumps(result)
                            })

                        if final_answer_found:
                            # Return final answer
                            processing_time = time.time() - start_time
                            return {
                                "answer": final_answer_found["data"]["answer"],
                                "reasoning": final_answer_found["data"].get("reasoning", ""),
                                "context_sources": final_answer_found["data"].get("context_sources", []),
                                "conversation_log": conversation_log,
                                "iterations": iteration,
                                "rlm_pattern": "true_rlm",
                                "processing_time_seconds": round(processing_time, 2)
                            }

                        # Continue with tool results
                        current_messages.append(message)
                        current_messages.extend(tool_results)

                        # Make next LM call (continue with Root LM system prompt)
                        response = await self.llm_client.chat(current_messages, {"tools": self.get_rlm_tools_schema()}, 200000, system_message["content"])

                    else:
                        # No tool calls, this is a direct response
                        conversation_log.append({
                            "role": "assistant",
                            "content": message.get("content", ""),
                            "timestamp": datetime.now().isoformat(),
                            "type": "direct_response"
                        })

                        processing_time = time.time() - start_time
                        return {
                            "answer": message.get("content", "No response generated"),
                            "reasoning": "Direct response without tool usage",
                            "context_sources": [],
                            "conversation_log": conversation_log,
                            "iterations": iteration,
                            "rlm_pattern": "true_rlm",
                            "processing_time_seconds": round(processing_time, 2)
                        }
                else:
                    break

            # Fallback if max iterations reached
            processing_time = time.time() - start_time
            return {
                "answer": "I apologize, but I was unable to process your request within the allowed steps. Please try rephrasing your question.",
                "reasoning": "Max iterations reached without final answer",
                "context_sources": [],
                "conversation_log": conversation_log,
                "iterations": iteration,
                "rlm_pattern": "true_rlm",
                "processing_time_seconds": round(processing_time, 2)
            }

        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "answer": f"An error occurred while processing your request: {str(e)}",
                "reasoning": "Error in RLM processing",
                "context_sources": [],
                "conversation_log": [],
                "iterations": 0,
                "rlm_pattern": "true_rlm",
                "processing_time_seconds": round(processing_time, 2),
                "error": str(e)
            }