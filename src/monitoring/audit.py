"""Audit logging system for agent actions and decisions."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import json
from loguru import logger


class AuditEventType(str, Enum):
    """Types of audit events."""
    AGENT_EXECUTION_START = "agent_execution_start"
    AGENT_EXECUTION_END = "agent_execution_end"
    TOOL_EXECUTION = "tool_execution"
    RAG_RETRIEVAL = "rag_retrieval"
    DECISION_MADE = "decision_made"
    CORRECTION_APPLIED = "correction_applied"
    HALLUCINATION_DETECTED = "hallucination_detected"
    ERROR_OCCURRED = "error_occurred"
    CONFIGURATION_CHANGED = "configuration_changed"


@dataclass
class AuditEvent:
    """An audit event record."""
    event_type: AuditEventType
    description: str
    agent_id: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["timestamp"] = self.timestamp.isoformat()
        return data


class AuditLogger:
    """Logs all agent actions for compliance and debugging."""
    
    def __init__(self, max_events: int = 10000):
        """Initialize audit logger."""
        self.max_events = max_events
        self.events: List[AuditEvent] = []
        logger.info(f"Initialized AuditLogger with max_events={max_events}")
    
    def log_event(
        self,
        event_type: AuditEventType,
        description: str,
        agent_id: str,
        session_id: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> AuditEvent:
        """Log an audit event."""
        event = AuditEvent(
            event_type=event_type,
            description=description,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            details=details or {},
            success=success,
            error_message=error_message,
        )
        
        self.events.append(event)
        
        # Maintain history limit
        if len(self.events) > self.max_events:
            self.events.pop(0)
        
        # Log to file
        log_level = "INFO" if success else "WARNING"
        logger.log(
            log_level,
            f"AUDIT: {event_type.value} - {description} (agent={agent_id}, session={session_id})"
        )
        
        return event
    
    def log_execution_start(
        self,
        agent_id: str,
        session_id: str,
        task: str,
        user_id: Optional[str] = None,
    ) -> AuditEvent:
        """Log start of agent execution."""
        return self.log_event(
            event_type=AuditEventType.AGENT_EXECUTION_START,
            description=f"Agent execution started",
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            details={"task": task[:200]},
        )
    
    def log_execution_end(
        self,
        agent_id: str,
        session_id: str,
        success: bool,
        execution_time: float,
        corrections_applied: int,
        result: Optional[str] = None,
    ) -> AuditEvent:
        """Log end of agent execution."""
        return self.log_event(
            event_type=AuditEventType.AGENT_EXECUTION_END,
            description=f"Agent execution ended - {'success' if success else 'failed'}",
            agent_id=agent_id,
            session_id=session_id,
            success=success,
            details={
                "execution_time": execution_time,
                "corrections_applied": corrections_applied,
                "result_preview": result[:100] if result else None,
            },
        )
    
    def log_tool_execution(
        self,
        agent_id: str,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> AuditEvent:
        """Log tool execution."""
        return self.log_event(
            event_type=AuditEventType.TOOL_EXECUTION,
            description=f"Executed tool: {tool_name}",
            agent_id=agent_id,
            session_id=session_id,
            success=success,
            error_message=error_message,
            details={
                "tool_name": tool_name,
                "arguments": str(arguments)[:200],
                "result_type": type(result).__name__,
            },
        )
    
    def log_rag_retrieval(
        self,
        agent_id: str,
        session_id: str,
        query: str,
        num_documents: int,
        success: bool = True,
    ) -> AuditEvent:
        """Log RAG document retrieval."""
        return self.log_event(
            event_type=AuditEventType.RAG_RETRIEVAL,
            description=f"Retrieved {num_documents} documents from knowledge base",
            agent_id=agent_id,
            session_id=session_id,
            success=success,
            details={
                "query": query[:150],
                "num_documents": num_documents,
            },
        )
    
    def log_correction(
        self,
        agent_id: str,
        session_id: str,
        issue: str,
        correction: str,
    ) -> AuditEvent:
        """Log correction applied."""
        return self.log_event(
            event_type=AuditEventType.CORRECTION_APPLIED,
            description=f"Self-correction applied",
            agent_id=agent_id,
            session_id=session_id,
            details={
                "issue": issue[:150],
                "correction": correction[:150],
            },
        )
    
    def log_hallucination(
        self,
        agent_id: str,
        session_id: str,
        risk_level: str,
        reasons: List[str],
    ) -> AuditEvent:
        """Log detected hallucination."""
        return self.log_event(
            event_type=AuditEventType.HALLUCINATION_DETECTED,
            description=f"Potential hallucination detected (risk: {risk_level})",
            agent_id=agent_id,
            session_id=session_id,
            details={
                "risk_level": risk_level,
                "reasons": reasons[:3],
            },
        )
    
    def get_events_for_session(self, session_id: str) -> List[AuditEvent]:
        """Get all events for a session."""
        return [e for e in self.events if e.session_id == session_id]
    
    def get_events_for_agent(
        self,
        agent_id: str,
        event_type: Optional[AuditEventType] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get events for an agent."""
        events = [
            e for e in self.events
            if e.agent_id == agent_id
            and (event_type is None or e.event_type == event_type)
        ]
        return [e.to_dict() for e in events[-limit:]]
    
    def get_session_timeline(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chronological timeline of events for a session."""
        events = self.get_events_for_session(session_id)
        events.sort(key=lambda e: e.timestamp)
        return [e.to_dict() for e in events]
    
    def get_error_audit(self, agent_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get all error events recent events for an agent."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        errors = [
            e for e in self.events
            if e.agent_id == agent_id
            and not e.success
            and e.timestamp >= cutoff
        ]
        return [e.to_dict() for e in errors[-50:]]
    
    def export_audit_log(self, agent_id: str, session_id: Optional[str] = None) -> str:
        """Export audit log as JSON."""
        if session_id:
            events = self.get_events_for_session(session_id)
        else:
            events = [e for e in self.events if e.agent_id == agent_id]
        
        return json.dumps([e.to_dict() for e in events], indent=2)


# Global audit logger
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
