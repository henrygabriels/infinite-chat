#!/usr/bin/env python3
"""
Simple Terminal Chat Client
Minimal interface that respects terminal theme and styling.
"""

import asyncio
import os
from typing import Optional
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleChatClient:
    """Simple terminal chat client that uses terminal defaults."""

    def __init__(self):
        # Fixed conversation IDs for two persistent universes
        self.default_conversation = "default-chat"
        self.alt_conversation = "alt-chat"

        # Load which conversation is currently active
        self.conversation_id = self._load_active_conversation()
        self.client = httpx.AsyncClient(timeout=120.0)

    def _load_active_conversation(self) -> str:
        """Load the active conversation ID from file, default to default conversation."""
        active_file = ".active_conversation"
        try:
            if os.path.exists(active_file):
                with open(active_file, 'r') as f:
                    active = f.read().strip()
                    return active if active in [self.default_conversation, self.alt_conversation] else self.default_conversation
        except Exception:
            pass
        return self.default_conversation

    def _save_active_conversation(self, conversation_id: str) -> None:
        """Save the active conversation ID to file."""
        active_file = ".active_conversation"
        try:
            with open(active_file, 'w') as f:
                f.write(conversation_id)
        except Exception:
            pass

    def _switch_conversation(self) -> str:
        """Switch to the other conversation and return confirmation message."""
        if self.conversation_id == self.default_conversation:
            self.conversation_id = self.alt_conversation
            name = "alt"
        else:
            self.conversation_id = self.default_conversation
            name = "default"

        self._save_active_conversation(self.conversation_id)
        return f"Switched to {name} conversation"

    async def run(self):
        """Run the chat client."""
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("                                         ")
        print("            infinite chat                 ")
        print("                                         ")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        if self.conversation_id == self.alt_conversation:
            print("(alt conversation active)")
        print("Type your message and press Enter to send.")
        print("Press Ctrl+C to quit.\n")

        try:
            while True:
                # Get user input
                try:
                    message = input("> ").strip()
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break

                if not message:
                    continue

                # Check for secret switch command
                if message == "/switch":
                    switch_msg = self._switch_conversation()
                    print(f"✨ {switch_msg}")
                    continue

                # Check for common typos of switch
                switch_typos = ["/swtich", "/swith", "/swicth", "/siwtch", "/swich", "/switchh", "/sswitch"]
                if message.lower() in switch_typos:
                    print("Learn to type, dummy.")
                    continue

                # Check for cheat codes
                if message.lower() == "/rosebud":
                    print("Yeah, typing messages to a chatbot will _definitely_ reel in the great green proverbial whale of that late great American dollar, money. You don't have to do any actual work. Just type the cheat codes and blammo, cashola, payday. Wonga. You utter moron.")
                    continue

                if message.lower() == "/cheese steak jimmy's":
                    print("Yeah, typing messages to a chatbot will _definitely_ reel in the great green proverbial whale of that late great American dollar, money. You don't have to do any actual work. Just type the cheat codes and blammo, cashola, payday. Wonga. You utter moron.")
                    continue

                if message.lower() == "/brat":
                    print("POP UP ADVERTS: NOW IN YOUR TERMINAL!")
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    print("GO BUY BRAT BY GABRIEL SMITH")
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    continue

                # Show typing indicator
                print("Thinking...", end="", flush=True)

                try:
                    # Send to API
                    response_text = await self._send_message(message)

                    # Clear typing indicator
                    print("\r" + " " * 20 + "\r", end="")

                    # Display response
                    print(f"\033[1mAssistant:\033[0m {response_text}")
                    print()

                except Exception as e:
                    # Clear typing indicator
                    print("\r" + " " * 20 + "\r", end="")
                    print(f"Error: {str(e)}")
                    print()

        finally:
            await self.client.aclose()

    async def _send_message(self, message: str) -> str:
        """Send message to API and return response."""
        url = "http://localhost:8421/api/chat"
        payload = {
            "message": message,
            "context_window_size": 200000
        }

        if self.conversation_id:
            payload["conversation_id"] = self.conversation_id

        response = await self.client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        self.conversation_id = data.get("conversation_id")

        return data["response"]

if __name__ == "__main__":
    client = SimpleChatClient()
    asyncio.run(client.run())