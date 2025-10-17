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
        self.rlm_mode = self._load_rlm_mode()  # Load RLM mode state
        self.show_agent_logs = self._load_agent_logs_mode()  # Load agent logs display mode
        self.client = httpx.AsyncClient(timeout=600.0)  # 10 minutes for True RLM processing

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

    def _load_rlm_mode(self) -> bool:
        """Load RLM mode state from file."""
        rlm_file = ".rlm_mode"
        try:
            if os.path.exists(rlm_file):
                with open(rlm_file, 'r') as f:
                    return f.read().strip().lower() == "true"
        except Exception:
            pass
        return False

    def _save_rlm_mode(self, rlm_mode: bool) -> None:
        """Save RLM mode state to file."""
        rlm_file = ".rlm_mode"
        try:
            with open(rlm_file, 'w') as f:
                f.write("true" if rlm_mode else "false")
        except Exception:
            pass

    def _load_agent_logs_mode(self) -> bool:
        """Load agent logs display mode from file."""
        logs_file = ".show_agent_logs"
        try:
            if os.path.exists(logs_file):
                with open(logs_file, 'r') as f:
                    return f.read().strip().lower() == "true"
        except Exception:
            pass
        return False

    def _save_agent_logs_mode(self, show_logs: bool) -> None:
        """Save agent logs display mode to file."""
        logs_file = ".show_agent_logs"
        try:
            with open(logs_file, 'w') as f:
                f.write("true" if show_logs else "false")
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

    async def _toggle_rlm_mode(self) -> str:
        """Toggle RLM mode and return confirmation message."""
        self.rlm_mode = not self.rlm_mode
        self._save_rlm_mode(self.rlm_mode)

        if self.rlm_mode:
            return "RLM Mode activated - Using Retrieval-Augmented Language Model for intelligent context retrieval"
        else:
            # When exiting RLM mode, migrate conversation history to standard storage
            try:
                url = f"http://localhost:8421/api/rlm-exit/{self.conversation_id}"
                response = await self.client.post(url, timeout=30.0)
                response.raise_for_status()
                result = response.json()

                if result.get("migrated", False):
                    return "RLM Mode deactivated - Conversation history migrated to standard storage"
                else:
                    return "RLM Mode deactivated - No history to migrate"
            except Exception:
                # Even if migration fails, still deactivate RLM mode
                return "RLM Mode deactivated - Note: History migration may have failed"

    def _toggle_agent_logs(self, show: bool) -> str:
        """Toggle agent logs display and return confirmation message."""
        self.show_agent_logs = show
        self._save_agent_logs_mode(show)

        if show:
            return "Agent logs display activated - Will show full RLM agent processing logs"
        else:
            return "Agent logs display hidden - Will hide RLM agent processing logs"

    def _show_help(self) -> None:
        """Display help information about available commands."""
        print("\n" + "="*60)
        print("INFINITE CHAT - AVAILABLE COMMANDS")
        print("="*60)

        print("\nConversation Management:")
        print("  /switch    - Switch between default and alt conversations")

        print("\nTrue RLM Mode (Retrieval-Augmented Language Model):")
        print("  /rlm       - Toggle True RLM mode on/off")
        print("             • Root LM strategically explores conversation context")
        print("             • Programmatic context access via tools")
        print("             • Recursive LM calls for deep analysis")

        print("\nAgent Logs (only available in True RLM mode):")
        print("  /view      - Show full Root LM processing logs")
        print("             • See strategic context exploration")
        print("             • View tool usage and recursive LM calls")
        print("  /hide      - Hide Root LM processing logs")

        print("\nHelp:")
        print("  /help      - Show this help message")

        print("\nUsage Tips:")
        print("  • Use /rlm to activate True RLM pattern")
        print("  • Use /view to see Root LM's strategic context exploration")
        print("  • Send complex queries to see recursive LM calls in action")
        print("  • Use /switch to maintain separate conversation contexts")
        print("  • Press Ctrl+C to quit")

        print("\n" + "="*60)
        print()

    async def _display_agent_logs(self) -> None:
        """Display the full RLM agent logs."""
        if not self.rlm_mode:
            print("Agent logs are only available in RLM mode")
            return

        try:
            url = f"http://localhost:8421/api/rlm-logs/{self.conversation_id}"
            response = await self.client.get(url, timeout=30.0)  # Increased timeout for logs retrieval
            response.raise_for_status()

            data = response.json()
            agent_logs = data.get('agent_logs', [])
            conversation_logs = data.get('conversation_logs', [])
            stats = data.get('stats', {})

            print("\n" + "="*80)
            print("RLM AGENT PROCESSING LOGS")
            print("="*80)

            if agent_logs:
                print(f"\nAgent Interactions ({len(agent_logs)} entries):")
                print("-" * 50)
                for i, log in enumerate(agent_logs, 1):
                    timestamp = log.get('timestamp', 'Unknown time')
                    role = log.get('role', 'Unknown')
                    content = log.get('content', '')
                    metadata = log.get('metadata', {})

                    print(f"\n{i}. [{timestamp[:19]}] {role.upper()}")
                    if metadata:
                        print(f"   Metadata: {metadata}")
                    print(f"   Content: {content[:200]}{'...' if len(content) > 200 else ''}")
            else:
                print("\nNo agent interactions recorded yet")

            print(f"\nRLM Stats:")
            print(f"   • RLM messages: {stats.get('rlm_messages_count', 0)}")
            print(f"   • Agent messages: {stats.get('agent_messages_count', 0)}")
            print(f"   • Estimated RLM tokens: {stats.get('estimated_rlm_tokens', 0)}")
            print(f"   • Estimated agent tokens: {stats.get('estimated_agent_tokens', 0)}")
            print(f"   • Mode active: {stats.get('mode_active', False)}")

            if conversation_logs:
                print(f"\nClean Conversation ({len(conversation_logs)} messages):")
                print("-" * 50)
                for i, log in enumerate(conversation_logs[-5:], len(conversation_logs)-4):  # Show last 5
                    role = log.get('role', 'Unknown')
                    content = log.get('content', '')
                    print(f"{i}. [{role.upper()}] {content[:100]}{'...' if len(content) > 100 else ''}")
            else:
                print("\nNo conversation messages yet")

            print("\n" + "="*80)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print("No RLM logs found for this conversation")
            else:
                print(f"Error retrieving agent logs: {e.response.status_code}")
        except Exception as e:
            print(f"Error retrieving agent logs: {str(e)}")

    async def run(self):
        """Run the chat client."""
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("                                         ")
        print("            infinite chat                 ")
        print("                                         ")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        if self.conversation_id == self.alt_conversation:
            print("(alt conversation active)")
        if self.rlm_mode:
            print("RLM Mode active - Enhanced context retrieval")
            if self.show_agent_logs:
                print("Agent logs display active")
        print("Type your message and press Enter to send.")
        print("Commands: /switch (change conversation), /rlm (toggle RLM mode), /view (show agent logs), /hide (hide agent logs), /help (show help)")
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
                    print(f"* {switch_msg}")
                    continue

                # Check for RLM mode toggle
                if message == "/rlm":
                    rlm_msg = await self._toggle_rlm_mode()
                    print(f"* {rlm_msg}")
                    continue

                # Check for agent logs viewing commands
                if message == "/view":
                    if not self.rlm_mode:
                        print("Agent logs are only available in RLM mode. Use '/rlm' to activate RLM mode first.")
                    else:
                        toggle_msg = self._toggle_agent_logs(True)
                        print(f"* {toggle_msg}")
                        await self._display_agent_logs()
                    continue

                if message == "/hide":
                    toggle_msg = self._toggle_agent_logs(False)
                    print(f"* {toggle_msg}")
                    continue

                # Check for help command
                if message == "/help":
                    self._show_help()
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

        # Choose endpoint based on RLM mode
        if self.rlm_mode:
            url = "http://localhost:8421/api/rlm-chat"
        else:
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

        # Show RLM stats if in RLM mode
        if self.rlm_mode and "rlm_stats" in data:
            rlm_stats = data["rlm_stats"]

            # Show processing time for True RLM
            if rlm_stats.get("rlm_pattern") == "true_rlm":
                processing_time = rlm_stats.get("processing_time_seconds", 0)
                iterations = rlm_stats.get("iterations_used", 0)
                if processing_time > 0:
                    print(f"Processing time: {processing_time:.1f}s ({iterations} iterations)")

            if rlm_stats.get("context_found", False):
                print(f"Found {rlm_stats.get('context_count', 0)} relevant context items")

            # Automatically show agent logs if enabled and in RLM mode
            if self.show_agent_logs:
                await self._display_agent_logs()

        return data["response"]

if __name__ == "__main__":
    client = SimpleChatClient()
    asyncio.run(client.run())