#!/usr/bin/env python3
"""
RLM Storage System
Handles separate storage for RLM mode conversations and agent interactions.
"""

import json
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime
from storage import ConversationStorage


class RLMStorage:
    """Storage system for RLM mode with separate logs for main dialogue and agent interactions."""

    def __init__(self, base_storage_dir: str = "conversations"):
        # Create separate directory structure for RLM conversations
        self.base_storage_dir = base_storage_dir
        self.rlm_dir = os.path.join(base_storage_dir, "rlm_mode")
        self.rlm_agent_dir = os.path.join(base_storage_dir, "rlm_agents")

        # Create directories if they don't exist
        os.makedirs(self.rlm_dir, exist_ok=True)
        os.makedirs(self.rlm_agent_dir, exist_ok=True)

        # Use standard storage for regular conversations
        self.standard_storage = ConversationStorage(base_storage_dir)

    def get_rlm_conversation_id(self, base_conversation_id: str) -> str:
        """Get RLM mode conversation ID for a base conversation."""
        return f"rlm_{base_conversation_id}"

    def get_rlm_agent_conversation_id(self, base_conversation_id: str) -> str:
        """Get RLM agent conversation ID for a base conversation."""
        return f"rlm_agent_{base_conversation_id}"

    def load_rlm_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load RLM mode conversation (clean user-assistant dialogue)."""
        rlm_conversation_id = self.get_rlm_conversation_id(conversation_id)
        filepath = os.path.join(self.rlm_dir, f"{rlm_conversation_id}.json")

        if not os.path.exists(filepath):
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_rlm_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """Save RLM mode conversation (clean user-assistant dialogue)."""
        rlm_conversation_id = self.get_rlm_conversation_id(conversation_id)
        filepath = os.path.join(self.rlm_dir, f"{rlm_conversation_id}.json")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)

    def load_rlm_agent_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load RLM agent conversation (agent interactions)."""
        agent_conversation_id = self.get_rlm_agent_conversation_id(conversation_id)
        filepath = os.path.join(self.rlm_agent_dir, f"{agent_conversation_id}.json")

        if not os.path.exists(filepath):
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_rlm_agent_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """Save RLM agent conversation (agent interactions)."""
        agent_conversation_id = self.get_rlm_agent_conversation_id(conversation_id)
        filepath = os.path.join(self.rlm_agent_dir, f"{agent_conversation_id}.json")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)

    def append_rlm_message(self, conversation_id: str, role: str, content: str) -> str:
        """Add message to RLM conversation and return message ID."""
        messages = self.load_rlm_conversation(conversation_id)

        message = {
            "id": f"rlm_{len(messages) + 1}_{datetime.now().strftime('%H%M%S')}",
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "conversation_mode": "rlm"
        }

        messages.append(message)
        self.save_rlm_conversation(conversation_id, messages)
        return message["id"]

    def append_rlm_agent_message(self, conversation_id: str, role: str, content: str,
                               metadata: Dict[str, Any] = None) -> str:
        """Add message to RLM agent conversation and return message ID."""
        messages = self.load_rlm_agent_conversation(conversation_id)

        message = {
            "id": f"agent_{len(messages) + 1}_{datetime.now().strftime('%H%M%S')}",
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        messages.append(message)
        self.save_rlm_agent_conversation(conversation_id, messages)
        return message["id"]

    def get_full_history_for_search(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get full conversation history for search (both RLM and standard)."""
        # For RLM mode, we want to search across all conversation history
        # including previous standard conversations if they exist

        # Try to load standard conversation first
        standard_messages = self.standard_storage.load_conversation(conversation_id)

        # Load RLM conversation
        rlm_messages = self.load_rlm_conversation(conversation_id)

        # Combine them, with RLM messages taking precedence for recent history
        # but include standard messages for comprehensive search
        all_messages = standard_messages + rlm_messages

        # Sort by timestamp to maintain chronological order
        all_messages.sort(key=lambda x: x.get('timestamp', ''))

        return all_messages

    def switch_to_rlm_mode(self, conversation_id: str) -> Tuple[str, str]:
        """Switch a conversation to RLM mode and return both conversation IDs."""
        rlm_conversation_id = self.get_rlm_conversation_id(conversation_id)
        agent_conversation_id = self.get_rlm_agent_conversation_id(conversation_id)

        # Initialize RLM conversation files if they don't exist
        if not os.path.exists(os.path.join(self.rlm_dir, f"{rlm_conversation_id}.json")):
            self.save_rlm_conversation(conversation_id, [])

        if not os.path.exists(os.path.join(self.rlm_agent_dir, f"{agent_conversation_id}.json")):
            self.save_rlm_agent_conversation(conversation_id, [])

        return rlm_conversation_id, agent_conversation_id

    def is_rlm_conversation(self, conversation_id: str) -> bool:
        """Check if a conversation is in RLM mode."""
        rlm_conversation_id = self.get_rlm_conversation_id(conversation_id)
        filepath = os.path.join(self.rlm_dir, f"{rlm_conversation_id}.json")
        return os.path.exists(filepath)

    def get_rlm_stats(self, conversation_id: str) -> Dict[str, Any]:
        """Get statistics about RLM conversation usage."""
        rlm_messages = self.load_rlm_conversation(conversation_id)
        agent_messages = self.load_rlm_agent_conversation(conversation_id)

        # Estimate tokens (rough approximation: ~4 characters per token)
        def estimate_tokens(text):
            return len(text) // 4 if text else 0

        return {
            "rlm_messages_count": len(rlm_messages),
            "agent_messages_count": len(agent_messages),
            "total_rlm_characters": sum(len(msg.get('content', '')) for msg in rlm_messages),
            "total_agent_characters": sum(len(msg.get('content', '')) for msg in agent_messages),
            "estimated_rlm_tokens": sum(estimate_tokens(msg.get('content', '')) for msg in rlm_messages),
            "estimated_agent_tokens": sum(estimate_tokens(msg.get('content', '')) for msg in agent_messages),
            "mode_active": self.is_rlm_conversation(conversation_id)
        }

    def migrate_from_rlm_mode(self, conversation_id: str) -> bool:
        """Migrate RLM conversation history to standard storage when exiting RLM mode."""
        rlm_messages = self.load_rlm_conversation(conversation_id)

        if not rlm_messages:
            return False  # No RLM history to migrate

        # Convert RLM messages to standard format and save to standard storage
        standard_messages = []
        for msg in rlm_messages:
            standard_msg = {
                "id": msg.get("id", f"migrated_{len(standard_messages)}"),
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", datetime.now().isoformat())
            }
            standard_messages.append(standard_msg)

        # Save to standard storage
        self.standard_storage.save_conversation(conversation_id, standard_messages)

        # Optionally, clean up RLM storage after successful migration
        # Commented out to preserve RLM logs in case user wants to switch back
        # self._cleanup_rlm_storage(conversation_id)

        return True

    def _cleanup_rlm_storage(self, conversation_id: str) -> None:
        """Clean up RLM storage files for a conversation."""
        rlm_conversation_id = self.get_rlm_conversation_id(conversation_id)
        agent_conversation_id = self.get_rlm_agent_conversation_id(conversation_id)

        rlm_file = os.path.join(self.rlm_dir, f"{rlm_conversation_id}.json")
        agent_file = os.path.join(self.rlm_agent_dir, f"{agent_conversation_id}.json")

        try:
            if os.path.exists(rlm_file):
                os.remove(rlm_file)
            if os.path.exists(agent_file):
                os.remove(agent_file)
        except Exception:
            pass  # Ignore cleanup errors

    def list_rlm_conversations(self) -> List[str]:
        """List all conversations that have RLM mode enabled."""
        if not os.path.exists(self.rlm_dir):
            return []

        files = os.listdir(self.rlm_dir)
        # Remove 'rlm_' prefix and '.json' suffix to get base conversation IDs
        return [f.replace('rlm_', '').replace('.json', '') for f in files if f.endswith('.json') and f.startswith('rlm_')]