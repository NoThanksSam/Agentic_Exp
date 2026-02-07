"""Integration tests for dashboard and monitoring components."""

import pytest
import asyncio
from src.agent import create_agent
from src.rag import RAGPipeline
from src.mcp import get_mcp_server
from src.monitoring import (
    get_metrics_collector,
    get_hallucination_detector,
    get_audit_logger,
    MetricType,
    AuditEventType,
)
from src.api.chat import get_chat_manager, ChatSession
from src.tools import create_tools


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class TestMonitoring:
    """Test monitoring components."""
    
    def test_metrics_collector(self):
        """Test metrics collection."""
        collector = get_metrics_collector()
        
        collector.record_metric(
            MetricType.EXECUTION_TIME,
            2.5,
            agent_id="test_agent"
        )
        
        stats = collector.get_stats("test_agent")
        assert stats.total_executions > 0
    
    def test_hallucination_detection(self):
        """Test hallucination detection."""
        detector = get_hallucination_detector()
        
        # Test with confident false statement
        result = detector.analyze(
            response="JavaScript was invented in 2010 by Google.",
            supporting_documents=["JavaScript was created in 1995 by Brendan Eich"]
        )
        
        assert result.is_hallucinated or result.risk_level != "none"
    
    def test_audit_logging(self):
        """Test audit logging."""
        logger = get_audit_logger()
        
        event = logger.log_event(
            event_type=AuditEventType.AGENT_EXECUTION_START,
            description="Test execution",
            agent_id="test_agent",
            session_id="test_session",
            user_id="test_user"
        )
        
        assert event is not None
        assert event.agent_id == "test_agent"
        
        events = logger.get_events_for_session("test_session")
        assert len(events) > 0


class TestChat:
    """Test chat functionality."""
    
    def test_chat_session_creation(self):
        """Test creating chat session."""
        manager = get_chat_manager()
        session = manager.create_session("test_agent")
        
        assert session is not None
        assert session.agent_id == "test_agent"
    
    def test_chat_messages(self):
        """Test adding messages to chat."""
        manager = get_chat_manager()
        session = manager.create_session("test_agent")
        
        user_msg = session.add_message("user", "Hello")
        assistant_msg = session.add_message("assistant", "Hi there!")
        
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"
    
    def test_conversation_context(self):
        """Test getting conversation context."""
        manager = get_chat_manager()
        session = manager.create_session("test_agent")
        
        session.add_message("user", "What is AI?")
        session.add_message("assistant", "AI is artificial intelligence")
        
        context = session.get_conversation_context(limit=5)
        assert "AI" in context
        assert "artificial intelligence" in context


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_interaction_flow(self):
        """Test full interaction flow with monitoring."""
        # Initialize components
        rag = RAGPipeline()
        mcp = get_mcp_server()
        metrics = get_metrics_collector()
        audit = get_audit_logger()
        
        # Add documents
        rag.add_text("Python is a programming language")
        
        # Create chat session
        chat_mgr = get_chat_manager()
        session = chat_mgr.create_session("test_agent")
        
        # Add messages
        session.add_message("user", "What is Python?")
        session.add_message("assistant", "Python is a programming language")
        
        # Record metrics
        metrics.record_execution(
            agent_id="test_agent",
            execution_time=1.5,
            success=True,
            confidence=0.95,
            corrections_applied=0,
            tools_used=1,
            session_id=session.session_id
        )
        
        # Log audit event
        audit.log_execution_end(
            agent_id="test_agent",
            session_id=session.session_id,
            success=True,
            execution_time=1.5,
            corrections_applied=0,
            result="Python is a programming language"
        )
        
        # Verify everything was recorded
        stats = metrics.get_stats("test_agent")
        assert stats.total_executions > 0
        
        events = audit.get_session_timeline(session.session_id)
        assert len(events) > 0
        
        assert len(session.messages) == 2


class TestDashboardEndpoints:
    """Test dashboard-related endpoints."""
    
    def test_mcp_tools_list(self):
        """Test getting tools list."""
        mcp = get_mcp_server()
        tools = mcp.get_all_tools()
        assert isinstance(tools, list)
    
    def test_audit_export(self):
        """Test exporting audit log."""
        audit = get_audit_logger()
        
        audit.log_event(
            event_type=AuditEventType.AGENT_EXECUTION_START,
            description="Test",
            agent_id="test_agent",
            session_id="test_session"
        )
        
        export = audit.export_audit_log("test_agent")
        assert isinstance(export, str)
        assert "test_agent" in export


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
