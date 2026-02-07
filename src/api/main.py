"""FastAPI application for the agentic framework."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import time
import uuid
import json
from loguru import logger
from datetime import datetime

from config.settings import settings
from src.utils import setup_logging
from src.agent import SelfCorrectingAgent, AgentResponse, create_agent
from src.rag import get_rag_pipeline
from src.mcp import get_mcp_server
from src.tools import create_tools, register_tools_with_mcp
from src.monitoring import (
    get_metrics_collector,
    get_hallucination_detector,
    get_audit_logger,
    AuditEventType,
)
from src.api.chat import get_chat_manager, ChatSession


# Setup logging
setup_logging(level=settings.log_level)


# Request/Response Models
class ExecuteAgentRequest(BaseModel):
    """Request to execute agent."""
    task: str
    context: Optional[str] = None
    max_iterations: Optional[int] = None
    include_reasoning: bool = True
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ExecuteAgentResponse(BaseModel):
    """Response from agent execution."""
    success: bool
    answer: str
    reasoning: Optional[str] = None
    steps_count: int
    corrections_applied: int
    execution_time_seconds: float
    hallucination: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    execution_id: str


class AddDocumentsRequest(BaseModel):
    """Request to add documents to RAG."""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = None


class RetrieveDocumentsRequest(BaseModel):
    """Request to retrieve documents."""
    query: str
    top_k: int = 5


class ExecuteToolRequest(BaseModel):
    """Request to execute a tool."""
    tool_name: str
    arguments: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Chat message request."""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat message response."""
    message_id: str
    response: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Global instances
agent: Optional[SelfCorrectingAgent] = None
rag_pipeline = None
mcp_server = None
metrics_collector = None
hallucination_detector = None
audit_logger = None
chat_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global agent, rag_pipeline, mcp_server, metrics_collector, hallucination_detector, audit_logger, chat_manager
    
    logger.info("Starting Agentic Framework API")
    
    # Initialize components
    rag_pipeline = get_rag_pipeline()
    mcp_server = get_mcp_server()
    metrics_collector = get_metrics_collector()
    hallucination_detector = get_hallucination_detector()
    audit_logger = get_audit_logger()
    chat_manager = get_chat_manager()
    
    # Register tools with MCP
    register_tools_with_mcp(mcp_server)
    
    # Create agent with tools
    tools = create_tools()
    agent = create_agent(
        name="MainAgent",
        tools=tools,
        model_name=settings.llm.model,
        temperature=settings.llm.temperature,
    )
    
    logger.info("Agentic Framework initialized successfully")
    
    yield
    
    logger.info("Shutting down Agentic Framework")


# Create FastAPI app
app = FastAPI(
    title=settings.api.project_name,
    description="Cohesive agentic framework with RAG, MCP, and self-correcting agents",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static dashboard
try:
    from pathlib import Path
    dashboard_path = Path(__file__).parent.parent.parent / "dashboard"
    if dashboard_path.exists():
        app.mount("/dashboard", StaticFiles(directory=str(dashboard_path), html=True), name="dashboard")
except Exception as e:
    logger.warning(f"Could not mount dashboard: {e}")


# Routes
@app.get("/")
async def root():
    """Root endpoint - redirect to dashboard."""
    return {
        "message": "Welcome to Agentic Framework",
        "version": "0.1.0",
        "links": {
            "dashboard": "http://localhost:8000/dashboard/",
            "docs": "http://localhost:8000/docs",
            "health": "http://localhost:8000/health/detailed",
        }
    }


@app.post("/agent/execute", response_model=ExecuteAgentResponse)
async def execute_agent(request: ExecuteAgentRequest):
    """Execute an agent to solve a task."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Generate IDs
    execution_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Executing agent task: {request.task}")
        
        # Log execution start
        audit_logger.log_execution_start(
            agent_id=agent.name,
            session_id=session_id,
            task=request.task,
            user_id=request.user_id,
        )
        
        start_time = time.time()
        
        # Get RAG context if needed
        context = request.context
        support_docs = None
        if request.context is None and rag_pipeline.documents:
            retrieved = await rag_pipeline.retrieve(request.task, k=3)
            if retrieved:
                context = rag_pipeline.get_context_from_documents(retrieved)
                support_docs = [doc.page_content for doc in retrieved]
                
                # Log RAG retrieval
                audit_logger.log_rag_retrieval(
                    agent_id=agent.name,
                    session_id=session_id,
                    query=request.task,
                    num_documents=len(retrieved),
                )
        
        # Run agent
        response: AgentResponse = await agent.run(
            task=request.task,
            context=context,
            max_iterations=request.max_iterations,
        )
        
        execution_time = time.time() - start_time
        
        # Detect hallucinations
        hallucination_info = hallucination_detector.analyze(
            response=response.answer,
            context=context,
            supporting_documents=support_docs,
        )
        
        if hallucination_info.is_hallucinated:
            audit_logger.log_hallucination(
                agent_id=agent.name,
                session_id=session_id,
                risk_level=hallucination_info.risk_level.value,
                reasons=hallucination_info.reasons,
            )
        
        # Record metrics
        metrics_collector.record_execution(
            agent_id=agent.name,
            execution_time=execution_time,
            success=response.success,
            confidence=1.0 - hallucination_info.confidence,
            corrections_applied=response.corrections_applied,
            tools_used=len(response.metadata.get("tools_used", 0)),
            session_id=session_id,
        )
        
        # Log execution end
        audit_logger.log_execution_end(
            agent_id=agent.name,
            session_id=session_id,
            success=response.success,
            execution_time=execution_time,
            corrections_applied=response.corrections_applied,
            result=response.answer,
        )
        
        return ExecuteAgentResponse(
            success=response.success,
            answer=response.answer,
            reasoning=response.reasoning if request.include_reasoning else None,
            steps_count=len(response.steps),
            corrections_applied=response.corrections_applied,
            execution_time_seconds=execution_time,
            hallucination=hallucination_info.to_dict() if hallucination_info.is_hallucinated else None,
            session_id=session_id,
            execution_id=execution_id,
        )
    except Exception as e:
        logger.error(f"Error executing agent: {e}")
        audit_logger.log_event(
            event_type=AuditEventType.ERROR_OCCURRED,
            description=f"Agent execution error: {str(e)}",
            agent_id=agent.name if agent else "unknown",
            session_id=session_id,
            success=False,
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/add-documents")
async def add_documents(request: AddDocumentsRequest):
    """Add documents to the RAG pipeline."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        metadata = request.metadata or {}
        if request.source:
            metadata["source"] = request.source
        
        rag_pipeline.add_text(request.content, metadata)
        
        logger.info(f"Added {len(request.content)} characters to RAG")
        
        return {
            "success": True,
            "message": "Documents added successfully",
            "content_length": len(request.content),
        }
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/retrieve")
async def retrieve_documents(request: RetrieveDocumentsRequest):
    """Retrieve documents from RAG."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        documents = await rag_pipeline.retrieve(request.query, k=request.top_k)
        
        results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]
        
        logger.info(f"Retrieved {len(results)} documents")
        
        return {
            "success": True,
            "query": request.query,
            "documents": results,
            "count": len(results),
        }
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/execute")
async def execute_tool(request: ExecuteToolRequest):
    """Execute a tool via MCP."""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    
    try:
        result = await mcp_server.execute_tool(
            request.tool_name,
            request.arguments,
        )
        
        return {
            "success": True,
            "tool": request.tool_name,
            "result": result,
        }
    except Exception as e:
        logger.error(f"Error executing tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/tools")
async def list_tools():
    """List all available tools from MCP server."""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    
    return {
        "success": True,
        "tools": mcp_server.get_all_tools(),
        "count": len(mcp_server.tools),
    }


@app.get("/mcp/resources")
async def list_resources():
    """List all available resources from MCP server."""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    
    return {
        "success": True,
        "resources": mcp_server.get_all_resources(),
        "count": len(mcp_server.resources),
    }


@app.get("/health/detailed")
async def health_detailed():
    """Get detailed health information."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "agent": "initialized" if agent else "not initialized",
            "rag": "initialized" if rag_pipeline else "not initialized",
            "mcp": "initialized" if mcp_server else "not initialized",
            "monitoring": "initialized" if metrics_collector else "not initialized",
        },
        "config": {
            "llm_provider": settings.llm.provider,
            "rag_vector_store": settings.rag.vector_store,
            "agent_max_iterations": settings.agent.max_iterations,
        },
    }


# ============ MONITORING & METRICS ENDPOINTS ============

@app.get("/metrics/stats")
async def get_metrics_stats(agent_id: str = "MainAgent"):
    """Get performance statistics for an agent."""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
    
    stats = metrics_collector.get_stats(agent_id)
    return {
        "agent_id": agent_id,
        "stats": stats.to_dict(),
    }


@app.get("/metrics/recent")
async def get_recent_metrics(agent_id: str = "MainAgent", limit: int = 100):
    """Get recent metrics."""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
    
    metrics = metrics_collector.get_recent_metrics(agent_id, limit)
    return {
        "agent_id": agent_id,
        "count": len(metrics),
        "metrics": metrics,
    }


@app.post("/metrics/reset")
async def reset_metrics(agent_id: Optional[str] = None):
    """Reset metrics for an agent."""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
    
    metrics_collector.reset_metrics(agent_id)
    return {
        "success": True,
        "message": f"Reset metrics for agent_id={agent_id or 'all'}",
    }


# ============ AUDIT LOG ENDPOINTS ============

@app.get("/audit/session/{session_id}")
async def get_session_audit(session_id: str):
    """Get audit trail for a session."""
    if not audit_logger:
        raise HTTPException(status_code=503, detail="Audit logger not initialized")
    
    timeline = audit_logger.get_session_timeline(session_id)
    return {
        "session_id": session_id,
        "event_count": len(timeline),
        "events": timeline,
    }


@app.get("/audit/agent/{agent_id}")
async def get_agent_audit(agent_id: str, event_type: Optional[str] = None, limit: int = 50):
    """Get audit events for an agent."""
    if not audit_logger:
        raise HTTPException(status_code=503, detail="Audit logger not initialized")
    
    try:
        from src.monitoring import AuditEventType
        parsed_type = AuditEventType(event_type) if event_type else None
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
    
    events = audit_logger.get_events_for_agent(agent_id, parsed_type, limit)
    return {
        "agent_id": agent_id,
        "event_count": len(events),
        "events": events,
    }


@app.get("/audit/errors/{agent_id}")
async def get_error_audit(agent_id: str, hours: int = 24):
    """Get error events for an agent."""
    if not audit_logger:
        raise HTTPException(status_code=503, detail="Audit logger not initialized")
    
    errors = audit_logger.get_error_audit(agent_id, hours)
    return {
        "agent_id": agent_id,
        "time_range": f"last {hours} hours",
        "error_count": len(errors),
        "errors": errors,
    }


@app.post("/audit/export/{agent_id}")
async def export_audit_log(agent_id: str, session_id: Optional[str] = None):
    """Export audit log as JSON."""
    if not audit_logger:
        raise HTTPException(status_code=503, detail="Audit logger not initialized")
    
    log_json = audit_logger.export_audit_log(agent_id, session_id)
    return {
        "data": log_json,
        "exported_at": datetime.utcnow().isoformat(),
    }


# ============ CHAT ENDPOINTS ============

@app.post("/chat/session")
async def create_chat_session(agent_id: str = "MainAgent") -> Dict[str, Any]:
    """Create a new chat session."""
    if not chat_manager:
        raise HTTPException(status_code=503, detail="Chat manager not initialized")
    
    session = chat_manager.create_session(agent_id)
    return {
        "session_id": session.session_id,
        "agent_id": session.agent_id,
        "created_at": session.created_at.isoformat(),
    }


@app.get("/chat/session/{session_id}")
async def get_chat_session(session_id: str) -> Dict[str, Any]:
    """Get chat session details."""
    if not chat_manager:
        raise HTTPException(status_code=503, detail="Chat manager not initialized")
    
    session = chat_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.to_dict()


@app.post("/chat/message")
async def send_chat_message(request: ChatRequest) -> ChatResponse:
    """Send a message and get agent response."""
    if not chat_manager or not agent:
        raise HTTPException(status_code=503, detail="Components not initialized")
    
    # Get or create session
    session_id = request.session_id
    if not session_id:
        session = chat_manager.create_session(agent.name)
        session_id = session.session_id
    else:
        session = chat_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    
    # Add user message
    user_msg = session.add_message("user", request.message)
    
    try:
        # Get conversation context
        context = session.get_conversation_context(limit=5)
        
        # Execute agent
        response = await agent.run(
            task=request.message,
            context=context if context else None,
            max_iterations=3,
        )
        
        # Add assistant response
        session.add_message("assistant", response.answer)
        
        # Record metrics
        metrics_collector.record_execution(
            agent_id=agent.name,
            execution_time=response.execution_time_seconds,
            success=response.success,
            corrections_applied=response.corrections_applied,
            tools_used=0,
            session_id=session_id,
        )
        
        # Log to audit
        audit_logger.log_event(
            event_type=AuditEventType.AGENT_EXECUTION_END,
            description="Chat message processed",
            agent_id=agent.name,
            session_id=session_id,
            success=response.success,
        )
        
        return ChatResponse(
            message_id=user_msg.id,
            response=response.answer,
            session_id=session_id,
        )
    
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        audit_logger.log_event(
            event_type=AuditEventType.ERROR_OCCURRED,
            description=f"Chat error: {str(e)}",
            agent_id=agent.name,
            session_id=session_id,
            success=False,
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/session/{session_id}/close")
async def close_chat_session(session_id: str):
    """Close a chat session."""
    if not chat_manager:
        raise HTTPException(status_code=503, detail="Chat manager not initialized")
    
    chat_manager.close_session(session_id)
    return {
        "success": True,
        "session_id": session_id,
        "closed_at": datetime.utcnow().isoformat(),
    }


@app.get("/chat/sessions")
async def list_chat_sessions(agent_id: Optional[str] = None):
    """List all chat sessions."""
    if not chat_manager:
        raise HTTPException(status_code=503, detail="Chat manager not initialized")
    
    sessions = chat_manager.get_all_sessions(agent_id)
    return {
        "agent_id": agent_id or "all",
        "count": len(sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "agent_id": s.agent_id,
                "created_at": s.created_at.isoformat(),
                "message_count": len(s.messages),
                "is_active": s.is_active,
            }
            for s in sessions
        ],
    }


# ============ WEBSOCKET CHAT ============

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    if not chat_manager or not agent:
        await websocket.close(code=1008, reason="Components not initialized")
        return
    
    # Get or validate session
    session = chat_manager.get_session(session_id)
    if not session:
        session = ChatSession(session_id, agent.name)
        chat_manager.sessions[session_id] = session
    
    # Connect websocket
    await websocket.accept()
    chat_manager.add_connection(session_id, websocket)
    
    logger.info(f"WebSocket connected for session: {session_id}")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            message_type = message.get("type", "chat")
            
            if message_type == "chat":
                user_message = message.get("message", "")
                if not user_message:
                    continue
                
                # Add user message
                user_msg = session.add_message("user", user_message)
                
                # Broadcast user message
                await chat_manager.broadcast_to_session(
                    session_id,
                    {
                        "type": "message",
                        "role": "user",
                        "content": user_message,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                
                # Get agent response
                try:
                    context = session.get_conversation_context(limit=5)
                    response = await agent.run(
                        task=user_message,
                        context=context if context else None,
                        max_iterations=3,
                    )
                    
                    # Add assistant message
                    assistant_msg = session.add_message("assistant", response.answer)
                    
                    # Broadcast agent response
                    await chat_manager.broadcast_to_session(
                        session_id,
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": response.answer,
                            "timestamp": datetime.utcnow().isoformat(),
                            "metadata": {
                                "corrections_applied": response.corrections_applied,
                                "execution_time": response.execution_time_seconds,
                            }
                        }
                    )
                
                except Exception as e:
                    logger.error(f"Error in WebSocket chat: {e}")
                    await chat_manager.broadcast_to_session(
                        session_id,
                        {
                            "type": "error",
                            "message": str(e),
                        }
                    )
            
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
        chat_manager.remove_connection(session_id, websocket)
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        chat_manager.remove_connection(session_id, websocket)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level=settings.log_level.lower(),
    )

