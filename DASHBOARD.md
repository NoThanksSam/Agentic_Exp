# Dashboard Guide

## Overview

The Agentic Framework includes a comprehensive real-time dashboard for monitoring, tracking, and interacting with agents. The dashboard provides:

- **Real-time Chat Interface**: Talk to agents in plain English
- **Live Metrics & Efficiency Tracking**: Monitor performance in real-time
- **Hallucination Detection**: Identify potentially false or unsupported claims
- **Audit Logging**: Complete audit trail of all actions
- **Session Management**: Track and manage multiple chat sessions

## Accessing the Dashboard

### Local Development
```bash
# Start the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Open dashboard in browser
http://localhost:8000/dashboard/
```

### With Docker
```bash
docker-compose up -d

# Dashboard available at
http://localhost:8000/dashboard/
```

## Dashboard Components

### 1. Chat Interface (Left Panel)
- **Purpose**: Real-time conversation with the agent
- **Features**:
  - Plain English input
  - Real-time WebSocket connection
  - Message history in current session
  - Automatic scrolling to latest message
  - Connection status indicator

**Usage**:
```
Type: "What is machine learning?"
Agent will respond using RAG knowledge base and self-correction
```

### 2. Metrics Panel (Top Right)
- **Executions**: Total number of agent executions
- **Success Rate**: Percentage of successful completions
- **Avg Time**: Average execution time in seconds
- **Corrections**: Total self-corrections applied

**Tabs**:
- **Metrics**: Current aggregated statistics
- **Performance**: Last execution details

### 3. Audit Log Panel (Bottom Left)
- **Color Coding**:
  - 🟢 Green: Successful operations
  - 🟡 Yellow: Warnings/Hallucinations
  - 🔴 Red: Errors

- **Hallucination Indicators**:
  - Risk Level shown: `HIGH`, `MEDIUM`, `LOW`, `NONE`
  - Can pinpoint specific problematic responses

**Example Events**:
```
AGENT_EXECUTION_START - Agent execution started
RAG_RETRIEVAL - Retrieved 3 documents from knowledge base
CORRECTION_APPLIED - Self-correction applied
HALLUCINATION_DETECTED ⚠️ MEDIUM - Potential hallucination detected
AGENT_EXECUTION_END - Agent execution ended - success
```

### 4. Session Info Panel (Bottom Right)
- **Session ID**: Unique identifier for current session
- **Status**: Active or closed
- **Message Count**: Total messages in session
- **Last Activity**: When last message was sent

## Features & Capabilities

### Real-time Monitoring

The dashboard updates metrics automatically every 5 seconds:
- Execution count
- Success rate
- Average response time
- Self-correction tracking
- Audit events

### Hallucination Detection

The framework detects potential hallucinations through multiple methods:

**Indicators Tracked**:
- High confidence language without supporting evidence
- Specific numbers without context
- Claims unsupported by RAG documents
- Internal contradictions

**Risk Levels**:
- `NONE` (0-25%): No hallucination signs
- `LOW` (25-50%): Minor concerns
- `MEDIUM` (50-75%): Significant concerns
- `HIGH` (75-100%): Strong likelihood of hallucination

**Display**: Look for ⚠️ badges in audit log

### Efficiency Tracking

**Metrics Shown**:
- **Success Rate**: % of tasks completed successfully
- **Avg Execution Time**: Average time to complete task
- **Self-Corrections**: How many refinements were needed
- **Tool Usage**: Number of tools invoked per execution

**Interpretation**:
- Higher success rate → Better agent tuning
- Lower execution time → More efficient prompting
- Lower corrections → Better initial responses

### AI Audit Trail

Every action is logged with:
- **Timestamp**: When action occurred
- **Event Type**: What happened
- **Description**: Details of action
- **Session ID**: Which session
- **Success Status**: Did it work?
- **Error Messages**: If failed, why?

**Exportable**: Download full audit log as JSON for compliance

## Typical Workflows

### 1. Simple Q&A
```
User: "Tell me about quantum computing"
↓
Agent: (searches RAG knowledge base)
↓
Agent: (provides answer with sources)
↓
Dashboard: Shows execution time, no hallucinations detected
```

### 2. Multi-turn Conversation
```
User 1: "What is AI?"
↓
Agent: (answers)
↓
User 2: (follow-up) "How does machine learning work?"
↓
Agent: (uses previous conversation as context)
↓
Continue...
```

### 3. Monitoring Efficiency
```
Watch metrics as you interact with agent
Success rate climbing:  0% → 25% → 50% → ...
Avg time decreasing: 5.2s → 4.1s → 3.3s → ...
Corrections dropping: 3 → 2 → 1 → 0
```

## API Integration

### REST API Examples

**Execute Agent (Alternative to Chat)**:
```bash
curl -X POST http://localhost:8000/agent/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Explain neural networks",
    "max_iterations": 5,
    "include_reasoning": true
  }'
```

**Check Metrics**:
```bash
curl http://localhost:8000/metrics/stats?agent_id=MainAgent
```

**Get Audit Events**:
```bash
curl http://localhost:8000/audit/agent/MainAgent?limit=20
```

**Export Audit Log**:
```bash
curl -X POST http://localhost:8000/audit/export/MainAgent \
  > audit_log.json
```

### WebSocket Usage (Advanced)

The dashboard uses WebSocket for real-time chat:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat/SESSION_ID');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'chat',
        message: 'Hello agent!'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.type); // "message" or "error"
    console.log(data.content); // Agent response
};
```

## Troubleshooting

### Dashboard Not Loading
1. Ensure API is running: `http://localhost:8000/health/detailed`
2. Check CORS is enabled in FastAPI
3. Clear browser cache

### No Messages Appearing
1. Check WebSocket connection in browser DevTools
2. Verify session was created: `GET /chat/session/{id}`
3. Check API logs for errors

### Metrics Not Updating
1. Execute an agent task first
2. Wait 5 seconds for automatic refresh
3. Or click a different tab and back

### Hallucinations Not Detected
1. Must have some RAG documents in knowledge base
2. Check /audit/agent/MainAgent for detection events
3. May need stronger hallucination on supported claims

## Performance Tuning

### For Better Metrics
1. **Faster Responses**: Reduce `AGENT_MAX_ITERATIONS`
2. **Higher Success Rate**: Improve RAG documents
3. **Fewer Corrections**: Refine system prompts
4. **Better Hallucination Detection**: Add more reference documents

### Dashboard Performance
- Dashboard uses WebSocket (efficient)
- Auto-refresh every 5 seconds (configurable)
- Message history limited per session
- Audit log limited to 1000 events

## Security Considerations

### For Production
1. **Authentication**: Add OAuth/JWT to endpoints
2. **Rate Limiting**: Limit requests per IP/session
3. **Input Validation**: Sanitize user input
4. **Audit Retention**: Implement log rotation
5. **CORS**: Restrict to specific domains

### Example Production Config
```python
# In config/settings.py
API_ORIGINS = ["https://yourdomain.com"]
RATE_LIMIT = "100/minute"
LOG_RETENTION = "90 days"
```

## Advanced Features

### Custom Monitoring
Add custom metrics via API:
```bash
curl -X POST http://localhost:8000/metrics/custom \
  -d '{"name": "custom_metric", "value": 42}'
```

### Session Export
Export full session with messages and audit trail:
```bash
curl http://localhost:8000/chat/sessions/{id} > session.json
```

### Compliance Reports
Generate audit reports for compliance:
```bash
curl http://localhost:8000/audit/export/MainAgent \
  > compliance_report.json
```

## See Also
- [README.md](README.md) - Main framework documentation
- [DASHBOARD_API.md](DASHBOARD_API.md) - Full API reference
- [examples/](examples/) - Example usage scripts
