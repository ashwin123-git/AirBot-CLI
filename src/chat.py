"""Chat session management for AirBot"""
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Message:
    """A chat message"""
    role: str
    content: str


@dataclass
class ChatSession:
    """Manages a chat session"""
    messages: List[Message] = field(default_factory=list)
    system_prompt: str = "You are AirBot, a helpful AI assistant."

    def add_user_message(self, content: str):
        """Add a user message to the session"""
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str):
        """Add an assistant message to the session"""
        self.messages.append(Message(role="assistant", content=content))

    def get_messages(self) -> List[dict]:
        """Get messages in format for Ollama"""
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        for msg in self.messages:
            msgs.append({"role": msg.role, "content": msg.content})
        return msgs

    def clear(self):
        """Clear chat history"""
        self.messages = []

    def get_history_text(self) -> str:
        """Get chat history as text"""
        if not self.messages:
            return ""

        lines = ["Chat history:"]
        for msg in self.messages:
            role = "You" if msg.role == "user" else "AirBot"
            lines.append(f"{role}: {msg.content[:100]}...")
        return "\n".join(lines)
