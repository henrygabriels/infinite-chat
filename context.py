from typing import List, Dict, Any

class ContextWindow:
    def __init__(self, max_tokens: int = 200000):
        self.max_tokens = max_tokens

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (characters / 4)."""
        return len(text) // 4

    def exact_token_count(self, text: str) -> int:
        """More precise token counting using words."""
        # Simple approximation: ~1.3 tokens per word
        words = len(text.split())
        chars = len(text)
        # Weight between character and word based estimation
        return int((chars / 4 + words * 1.3) / 2)

    def calculate_message_tokens(self, message: Dict[str, Any]) -> int:
        """Calculate tokens for a message including metadata."""
        # Include role and content in token count
        content = f"{message['role']}: {message['content']}"
        return self.exact_token_count(content)

    def get_context_window(self, messages: List[Dict[str, Any]], reserve_tokens: int = 20000) -> List[Dict[str, Any]]:
        """
        Get sliding window of messages within token limit.
        Reserve space for system prompt and potential search results.
        """
        if not messages:
            return []

        available_tokens = self.max_tokens - reserve_tokens
        window_messages = []
        used_tokens = 0

        # Start from the end (most recent messages)
        for message in reversed(messages):
            message_tokens = self.calculate_message_tokens(message)

            if used_tokens + message_tokens <= available_tokens:
                window_messages.insert(0, message)
                used_tokens += message_tokens
            else:
                break

        return window_messages

    def can_fit_message(self, messages: List[Dict[str, Any]], new_content: str, reserve_tokens: int = 20000) -> bool:
        """Check if a new message can fit in the context window."""
        available_tokens = self.max_tokens - reserve_tokens

        # Calculate current usage
        current_tokens = sum(self.calculate_message_tokens(msg) for msg in messages)

        # Add new message tokens
        new_tokens = self.exact_token_count(f"user: {new_content}")

        return (current_tokens + new_tokens) <= available_tokens

    def get_window_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get statistics about the current context window."""
        if not messages:
            return {"total_messages": 0, "total_tokens": 0, "average_tokens_per_message": 0}

        total_tokens = sum(self.calculate_message_tokens(msg) for msg in messages)
        avg_tokens = total_tokens // len(messages)

        return {
            "total_messages": len(messages),
            "total_tokens": total_tokens,
            "average_tokens_per_message": avg_tokens
        }