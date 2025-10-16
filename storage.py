import json
import os
from typing import List, Dict, Any
from datetime import datetime
import uuid

class ConversationStorage:
    def __init__(self, storage_dir: str = "conversations"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def load_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load conversation from JSON file."""
        filepath = os.path.join(self.storage_dir, f"{conversation_id}.json")
        if not os.path.exists(filepath):
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """Save conversation to JSON file."""
        filepath = os.path.join(self.storage_dir, f"{conversation_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)

    def append_message(self, conversation_id: str, role: str, content: str) -> str:
        """Add message to conversation and return message ID."""
        messages = self.load_conversation(conversation_id)

        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        messages.append(message)
        self.save_conversation(conversation_id, messages)
        return message["id"]

    def get_message_by_id(self, conversation_id: str, message_id: str) -> Dict[str, Any]:
        """Get specific message by ID."""
        messages = self.load_conversation(conversation_id)
        for msg in messages:
            if msg["id"] == message_id:
                return msg
        return None

    def get_message_index(self, conversation_id: str, message_id: str) -> int:
        """Get index of message in conversation."""
        messages = self.load_conversation(conversation_id)
        for i, msg in enumerate(messages):
            if msg["id"] == message_id:
                return i
        return -1

    def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        if not os.path.exists(self.storage_dir):
            return []

        files = os.listdir(self.storage_dir)
        return [f.replace('.json', '') for f in files if f.endswith('.json')]