"""WebSocket chat handler for real-time agent interaction."""

from typing import Dict, Set, Optional, Any
import asyncio
import json
from datetime import datetime
from loguru import logger
from pydantic import BaseModel
import uuid


class ChatMessage(BaseModel):
    """A chat message."""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    session_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
        }


class ChatSession:
    """Manages a single chat session with an agent."""
    
    def __init__(self, session_id: str, agent_id: str = "default"):
        """Initialize chat session."""
        self.session_id = session_id
        self.agent_id = agent_id
        self.messages: list[ChatMessage] = []
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.is_active = True
        logger.info(f"Created chat session: {session_id}")
    
    def add_message(self, role: str, content: str) -> ChatMessage:
        """Add message to session."""
        message = ChatMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            session_id=self.session_id,
        )
        self.messages.append(message)
        self.last_activity = datetime.utcnow()
        return message
    
    def get_conversation_context(self, limit: int = 10) -> str:
        """Get recent conversation as context string."""
        recent = self.messages[-limit:]
        context = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in recent
        ])
        return context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": len(self.messages),
            "is_active": self.is_active,
            "messages": [msg.to_dict() for msg in self.messages],
        }


class ChatManager:
    """Manages multiple chat sessions."""
    
    def __init__(self):
        """Initialize chat manager."""
        self.sessions: Dict[str, ChatSession] = {}
        self.connections: Dict[str, Set[Any]] = {}  # session_id -> set of connections
        logger.info("Initialized ChatManager")
    
    def create_session(self, agent_id: str = "default") -> ChatSession:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id, agent_id)
        self.sessions[session_id] = session
        self.connections[session_id] = set()
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session."""
        return self.sessions.get(session_id)
    
    def add_connection(self, session_id: str, connection: Any) -> None:
        """Add a WebSocket connection to a session."""
        if session_id not in self.connections:
            self.connections[session_id] = set()
        self.connections[session_id].add(connection)
        logger.info(f"Added connection to session {session_id}")
    
    def remove_connection(self, session_id: str, connection: Any) -> None:
        """Remove a WebSocket connection from a session."""
        if session_id in self.connections:
            self.connections[session_id].discard(connection)
            logger.info(f"Removed connection from session {session_id}")
    
    async def broadcast_to_session(
        self,
        session_id: str,
        message: Dict[str, Any],
    ) -> None:
        """Broadcast message to all connections in a session."""
        if session_id not in self.connections:
            return
        
        payload = json.dumps(message)
        disconnected = set()
        
        for connection in self.connections[session_id]:
            try:
                await connection.send_text(payload)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected connections
        self.connections[session_id] -= disconnected
    
    def get_all_sessions(self, agent_id: Optional[str] = None) -> list[ChatSession]:
        """Get all sessions, optionally filtered by agent."""
        if agent_id:
            return [s for s in self.sessions.values() if s.agent_id == agent_id]
        return list(self.sessions.values())
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a session."""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session_id,
            "agent_id": session.agent_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": len(session.messages),
            "is_active": session.is_active,
        }
    
    def close_session(self, session_id: str) -> None:
        """Close a chat session."""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            logger.info(f"Closed session {session_id}")


# Global chat manager
_chat_manager: Optional[ChatManager] = None


def get_chat_manager() -> ChatManager:
    """Get or create chat manager."""
    global _chat_manager
    if _chat_manager is None:
        _chat_manager = ChatManager()
    return _chat_manager
