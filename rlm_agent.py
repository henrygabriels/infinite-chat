#!/usr/bin/env python3
"""
RLM (Retrieval-Augmented Language Model) Agent
Context retrieval agent that searches conversation history and enriches user messages.
"""

import json
from typing import List, Dict, Any, Optional
from llm import LLMClient
from storage import ConversationStorage
from search import FuzzySearch


class RLMAgent:
    """Agent that retrieves relevant context from conversation history."""

    def __init__(self, llm_client: LLMClient, storage: ConversationStorage, search: FuzzySearch):
        self.llm_client = llm_client
        self.storage = storage
        self.search = search

    def get_rlm_system_prompt(self) -> str:
        """Get the system prompt for the RLM context retrieval agent."""
        return """You are a context retrieval agent for a conversation system. Your job is to find relevant historical context that might help the assistant answer the user's message.

THE USER'S MESSAGE THAT YOU SHOULD REPLY TO WAS: "{user_message}"

Reply to the user directly. In chat history, I found the following message-response pairs which may be of relevance, but are not necessarily relevant:

{relevant_context}

If these historical exchanges are genuinely useful and relevant to the user's question, reference them naturally in your response. If they are not relevant or helpful, do not reference them at all.

If you need more specific historical context to better answer the user's question, use the search tool to find more relevant conversations. When using the search tool, respond to ME (the system), not the user - I will take your findings and incorporate them into the final response.

Your response should be helpful and direct as if you are the assistant, but with the benefit of historical context when relevant."""

    async def retrieve_context(self, conversation_id: str, user_message: str,
                             context_limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context from conversation history."""
        # Load conversation history (use full history for search)
        if hasattr(self.storage, 'get_full_history_for_search'):
            messages = self.storage.get_full_history_for_search(conversation_id)
        else:
            messages = self.storage.load_conversation(conversation_id)

        if not messages:
            return []

        # Search for relevant messages
        search_results = self.search.search_messages(messages, user_message, context_limit)

        # For each search result, try to get the message pair (user-assistant)
        context_pairs = []
        for result in search_results:
            message_id = result['message_id']

            # Find the message and its pair
            context_messages = self.search.expand_context(
                messages, message_id, "both", pairs=1
            )

            # Only add if we have a meaningful exchange
            if len(context_messages) >= 2:
                context_pairs.extend(context_messages)

        # Remove duplicates and limit context
        unique_context = []
        seen_ids = set()
        for msg in context_pairs:
            if msg['id'] not in seen_ids:
                unique_context.append(msg)
                seen_ids.add(msg['id'])

        return unique_context[:10]  # Limit to prevent context overflow

    def format_context(self, context_messages: List[Dict[str, Any]]) -> str:
        """Format context messages for inclusion in the prompt."""
        if not context_messages:
            return "No relevant historical context found."

        formatted = []
        for msg in context_messages:
            role = msg['role'].upper()
            content = msg['content']
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            formatted.append(f"{role}: {content}")

        return "\n\n".join(formatted)

    async def process_user_message(self, conversation_id: str, user_message: str) -> Dict[str, Any]:
        """Process user message through RLM agent to get enriched context."""
        # Step 1: Retrieve relevant context
        context_messages = await self.retrieve_context(conversation_id, user_message)

        # Step 2: Prepare the enriched prompt
        formatted_context = self.format_context(context_messages)
        system_prompt = self.get_rlm_system_prompt().format(
            user_message=user_message,
            relevant_context=formatted_context
        )

        # Step 3: Send to LLM for context processing if there's substantial context
        if context_messages and len(context_messages) > 0:
            try:
                # Create messages for RLM agent
                agent_messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]

                # Get RLM agent response
                response = await self.llm_client.chat(
                    agent_messages,
                    {},
                    context_window_size=200000
                )

                agent_response = response["choices"][0]["message"]["content"]

                return {
                    "original_message": user_message,
                    "context_messages": context_messages,
                    "enriched_prompt": agent_response,
                    "has_context": True
                }

            except Exception as e:
                # Fallback if RLM agent fails
                print(f"RLM agent processing failed: {e}")

        # Fallback: return original message with basic context
        return {
            "original_message": user_message,
            "context_messages": context_messages,
            "enriched_prompt": user_message,  # Use original message as fallback
            "has_context": len(context_messages) > 0
        }

    def create_assistant_prompt(self, rlm_result: Dict[str, Any]) -> str:
        """Create the final prompt for the assistant model."""
        if rlm_result["has_context"]:
            # Use the RLM agent's enriched response
            return rlm_result["enriched_prompt"]
        else:
            # Use original message if no context found
            return rlm_result["original_message"]