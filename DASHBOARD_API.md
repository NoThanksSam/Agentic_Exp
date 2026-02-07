"""API endpoints for the dashboard and monitoring."""

# MAIN DASHBOARD
GET /dashboard/
    - Serves the main dashboard UI
    - Real-time chat interface
    - Live metrics and monitoring
    - Audit log viewer
    - Hallucination detection display

# CHAT ENDPOINTS
POST /chat/session
    - Create a new chat session
    - Returns: session_id, agent_id, created_at
    
GET /chat/session/{session_id}
    - Get chat session details
    - Returns: session data with all messages
    
POST /chat/message
    - Send message to agent and get response
    - Body: {"message": "...", "session_id": "..."}
    - Returns: response, session_id, metadata
    
POST /chat/session/{session_id}/close
    - Close a chat session
    
GET /chat/sessions
    - List all chat sessions
    - Optional: ?agent_id=MainAgent

# WEBSOCKET CHAT (Real-time)
WS /ws/chat/{session_id}
    - WebSocket endpoint for real-time chat
    - Send: {"type": "chat", "message": "..."}
    - Send: {"type": "ping"}
    - Receive: {"type": "message", "role": "user|assistant", "content": "..."}

# METRICS & PERFORMANCE
GET /metrics/stats?agent_id=MainAgent
    - Get performance statistics
    - Returns: total_executions, success_rate, avg_execution_time, etc.
    
GET /metrics/recent?agent_id=MainAgent&limit=100
    - Get recent metrics
    - Returns: list of metric data points
    
POST /metrics/reset?agent_id=MainAgent
    - Reset metrics

# AUDIT LOG
GET /audit/session/{session_id}
    - Get audit trail for a session
    - Returns: chronological timeline of events
    
GET /audit/agent/{agent_id}?event_type=...&limit=50
    - Get audit events for an agent
    - Supports filtering by event_type
    
GET /audit/errors/{agent_id}?hours=24
    - Get error events for agent
    
POST /audit/export/{agent_id}?session_id=...
    - Export audit log as JSON

# HALLUCINATION DETECTION
- Integrated into /agent/execute response
- Response includes: hallucination.is_hallucinated, risk_level, reasons
- Risk levels: none, low, medium, high
- Displayed in audit log with special highlighting

# MONITORING
GET /health/detailed
    - Detailed health information
    - Component status (agent, rag, mcp, monitoring)
    - Configuration details
    
GET /mcp/tools
    - List available tools
    
GET /mcp/resources
    - List available resources
