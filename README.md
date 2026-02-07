# Agentic Framework

A cohesive, production-ready framework for building intelligent agents with Retrieval-Augmented Generation (RAG), Model Context Protocol (MCP), self-correcting capabilities, and **real-time dashboard UI**.

## 🎯 Features

### 🤖 Self-Correcting Agents
- Built-in reflection and self-correction mechanisms
- Multi-iteration refinement with intelligent feedback
- Confidence scoring and validation
- Tool execution with error handling and recovery

### 🔍 Retrieval-Augmented Generation (RAG)
- Multiple vector store backends (ChromaDB, Pinecone, Weaviate)
- Intelligent document chunking and embedding
- Semantic search with similarity filtering
- Context augmentation for LLM responses

### 🔌 Model Context Protocol (MCP)
- Standardized tool and resource management
- Extensible tool registration system
- Prompt template management
- Resource-based context passing

### 📊 Real-time Dashboard
- **Live Chat Interface**: Talk to agents in plain English
- **Performance Metrics**: Track efficiency and success rates
- **Hallucination Detection**: Identify suspicious/unfounded claims
- **Audit Logging**: Complete compliance trail
- **WebSocket Support**: Real-time updates

### 🛠️ Built-in Tools
- **Document Retrieval**: Access knowledge base via RAG
- **Calculator**: Evaluate mathematical expressions
- **JSON Processor**: Validate and transform JSON
- **Memory System**: Store and retrieve agent memory
- **Web Search**: (Mock implementation, ready for integration)

### ⚡ FastAPI Integration
- RESTful API for agent execution
- Real-time document ingestion
- Tool execution endpoints
- WebSocket chat endpoints
- Health monitoring and detailed diagnostics

### 🐳 Docker Support
- Production-ready containerization
- Docker Compose for multi-service setup
- Volume management for persistence

## 🎨 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Dashboard UI (Real-time)                  │
│  (Chat, Metrics, Hallucination Detection, Audit Logs)      │
└────────────────┬────────────────────────────────────────────┘
                 │
            ┌────┴─────────────────────┐
            │                           │
        REST API                    WebSocket
            │                           │
        ┌───▼──────────────────────────▼──┐
        │          FastAPI Server          │
        │  (Agent Execution, Monitoring)  │
        └────────────┬────────────────────┘
                     │
    ┌────────────────┼────────────────────┐
    │                │                    │
    ▼                ▼                    ▼
┌─────────┐ ┌──────────────┐ ┌──────────────────┐
│  Agent  │ │     RAG      │ │  Monitoring &    │
│ Engine  │ │   Pipeline   │ │     Auditing     │
└─┬───────┘ └────┬─────────┘ └──────┬───────────┘
  │              │                  │
  ├──────────────┴──────────────────┤
  │                                 │
  ▼                                 ▼
┌──────────────┐      ┌──────────────────────┐
│ LLM Provider │      │   Vector Store &     │
│ (OpenAI...)  │      │  Monitoring DB       │
└──────────────┘      └──────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional)
- OpenAI API key

### Installation

1. **Clone and setup**
```bash
cd /workspaces/Agentic_Exp
cp .env.example .env
# Edit .env with your OpenAI API key
```

2. **Install dependencies**
```bash
pip install -e .
```

3. **Configure environment**
```bash
export OPENAI_API_KEY="sk_..."
export LLM_MODEL="gpt-4-turbo-preview"
```

### Running Locally

**Start the API server:**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Access the dashboard:**
- Dashboard UI: http://localhost:8000/dashboard/
- API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Running with Docker

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

## 📊 Dashboard Usage

### Real-time Chat Interface
```
Simply type in plain English:
"What is machine learning?"
"Explain quantum computing"
"What are the benefits of AI?"

Agent responds with:
✓ RAG-augmented answers
✓ Self-corrected responses
✓ Hallucination detection
✓ Execution metrics
```

### Monitoring Metrics
- **Executions**: Total number of agent runs
- **Success Rate**: Percentage of successful completions
- **Avg Time**: Average execution time
- **Corrections**: Self-corrections applied

### Hallucination Detection
Risk levels displayed in audit log:
- 🟢 **NONE**: No concerns
- 🟡 **LOW**: Minor concerns
- 🟠 **MEDIUM**: Significant concerns  
- 🔴 **HIGH**: Strong hallucination likelihood

### Audit Trail
Complete log of:
- Agent executions
- Document retrievals
- Tools executed
- Corrections applied
- Errors occurred
- Hallucinations detected

**Export**: Download audit log as JSON for compliance

## 💬 ChatInterface Examples

### Example 1: Simple Q&A
```
User: "What is Python?"
↓
Agent: (Searches RAG knowledge base)
↓
Agent: "Python is a high-level programming language..."
↓
Dashboard: Displays execution time, success rate, no hallucinations
```

### Example 2: Multi-turn Conversation
```
User 1: "What is machine learning?"
↓
Agent: (Answers with RAG context)
↓
User 2: "How does it differ from deep learning?"
↓
Agent: (Uses conversation history for context)
↓
Continue building knowledge...
```

### Example 3: Monitoring Efficiency
```
Watch dashboard in real-time:
Success rate:  0% → 25% → 50% → 75%
Avg time:      5.2s → 4.1s → 3.3s → 2.8s
Corrections:   3 → 2 → 1 → 0
```

## 🔧 Configuration

Edit `.env` or set environment variables:

```bash
# LLM
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.7

# RAG
RAG_VECTOR_STORE=chromadb
RAG_CHUNK_SIZE=1000
RAG_MAX_RETRIEVED_DOCS=5

# Agent
AGENT_MAX_ITERATIONS=10
AGENT_ENABLE_SELF_CORRECTION=true

# API
API_HOST=0.0.0.0
API_PORT=8000

# Dashboard
DASHBOARD_AUTO_REFRESH=5000  # milliseconds
```

## 📁 Project Structure

```
agentic-framework/
├── src/
│   ├── agent/           # Self-correcting agent engine
│   ├── rag/            # RAG pipeline
│   ├── mcp/            # MCP server
│   ├── tools/          # Built-in tools
│   ├── api/            # FastAPI application
│   ├── monitoring/     # Metrics, hallucination detection, auditing
│   └── utils.py        # Utility functions
├── config/
│   └── settings.py     # Configuration management
├── dashboard/          # Web UI (HTML/JS/CSS)
│   └── index.html      # Dashboard interface
├── examples/           # Usage examples
├── tests/             # Test suite
├── Dockerfile         # Docker image
├── docker-compose.yml # Multi-service setup
├── pyproject.toml    # Dependencies
└── README.md
```

## 🌐 API Endpoints

### Chat
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/chat/session` | Create chat session |
| GET | `/chat/session/{id}` | Get session details |
| POST | `/chat/message` | Send message |
| POST | `/chat/session/{id}/close` | Close session |
| GET | `/chat/sessions` | List all sessions |
| WS | `/ws/chat/{id}` | WebSocket real-time chat |

### Metrics
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/metrics/stats` | Get performance stats |
| GET | `/metrics/recent` | Get recent metrics |
| POST | `/metrics/reset` | Reset metrics |

### Audit
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/audit/session/{id}` | Session audit trail |
| GET | `/audit/agent/{id}` | Agent audit events |
| GET | `/audit/errors/{id}` | Error events |
| POST | `/audit/export/{id}` | Export audit log |

### Agent & RAG
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/agent/execute` | Run agent on task |
| POST | `/rag/add-documents` | Add to knowledge base |
| POST | `/rag/retrieve` | Query knowledge base |

### Tools & Monitoring
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/tools/execute` | Execute a tool |
| GET | `/mcp/tools` | List tools |
| GET | `/health/detailed` | Health info |

## 🧪 Development

### Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/

# Lint
ruff check src/

# Type checking
mypy src/
```

### Running Examples

```bash
python examples/basic_agent.py
python examples/rag_demo.py
python examples/mcp_demo.py
```

## 🎓 Advanced Usage

### Custom Tools

```python
from src.mcp import get_mcp_server

mcp = get_mcp_server()

async def my_tool(query: str) -> str:
    return f"Result for {query}"

mcp.register_tool(
    name="my_tool",
    description="My custom tool",
    handler=my_tool,
    input_schema={"query": "str"}
)
```

### Custom RAG Sources

```python
rag = get_rag_pipeline()

# From text
rag.add_text("Some text content")

# From directory
rag.add_documents_from_directory("./documents", "**/*.pdf")
```

### Monitoring Custom Metrics

```python
from src.monitoring import get_metrics_collector, MetricType

metrics = get_metrics_collector()
metrics.record_metric(
    MetricType.CUSTOM,
    value=42.5,
    agent_id="MyAgent",
    metadata={"custom_field": "value"}
)
```

## 📈 Hallucination Detection

The framework automatically detects suspicious claims through:

1. **High Confidence Language** - Confident assertions without evidence
2. **Unsupported Numbers** - Specific figures without context
3. **Document Mismatch** - Claims not in RAG documents
4. **Contradictions** - Internal inconsistencies
5. **Vague Language** - Excessive uncertainty markers

**Example Detection**:
```
Input: "Python was invented in 2000"
Detection: HIGH risk - specific date unsupported, contradicts RAG docs
Display: ⚠️ HIGH - Suspicious claim detected in audit log
```

## 🔒 Security

### For Production
1. **Authentication**: Add OAuth/JWT
2. **Rate Limiting**: Limit requests
3. **CORS**: Restrict domains
4. **Input Validation**: Sanitize input
5. **Audit Retention**: Implement log rotation

### Example Config
```python
# In config/settings.py
API_ALLOWED_ORIGINS = ["https://yourdomain.com"]
RATE_LIMIT_PER_MINUTE = 100
AUDIT_LOG_RETENTION_DAYS = 90
```

## 📚 Documentation

- [Dashboard Guide](DASHBOARD.md) - Complete dashboard documentation
- [API Reference](DASHBOARD_API.md) - Full API endpoint details
- [Examples](examples/) - Working code examples
- [Tests](tests/) - Test examples

## 🚨 Troubleshooting

### Dashboard Not Loading
```bash
# Check API is running
curl http://localhost:8000/health/detailed

# Check logs
tail -f logs/agentic_framework.log
```

### No Chat Messages
- Verify WebSocket: Check browser DevTools → Network → WS
- Check session exists: `GET /chat/session/{id}`
- Review API errors

### Metrics Not Updating
- Execute an agent task first
- Dashboard auto-refreshes every 5 seconds
- Manually refresh browser

### Hallucinations Not Detected
- Add RAG documents first
- Check `/audit/agent/MainAgent` for detection logs
- May need stronger foundational claims

## 🗺️ Roadmap

- [ ] Multi-agent chat interface
- [ ] Data visualization (charts/graphs)
- [ ] Real-time collaboration
- [ ] Advanced prompt management
- [ ] Web UI dashboard (React)
- [ ] Mobile app support
- [ ] Integration with more LLM providers

## 📝 License

MIT License - see LICENSE file

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes and test
4. Submit pull request

## 📞 Support

For issues or questions:
- Check [DASHBOARD.md](DASHBOARD.md) for usage
- Review [examples/](examples/) for code
- Check [tests/](tests/) for patterns
- Review logs in `logs/agentic_framework.log`

## 🔗 References

- [LangChain Documentation](https://docs.langchain.com)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [OpenAI API](https://platform.openai.com/docs)
- [ChromaDB](https://docs.trychroma.com)
- Built-in reflection and self-correction mechanisms
- Multi-iteration refinement with intelligent feedback
- Confidence scoring and validation
- Tool execution with error handling and recovery

### 🔍 Retrieval-Augmented Generation (RAG)
- Multiple vector store backends (ChromaDB, Pinecone, Weaviate)
- Intelligent document chunking and embedding
- Semantic search with similarity filtering
- Context augmentation for LLM responses

### 🔌 Model Context Protocol (MCP)
- Standardized tool and resource management
- Extensible tool registration system
- Prompt template management
- Resource-based context passing

### 🛠️ Built-in Tools
- **Document Retrieval**: Access knowledge base via RAG
- **Calculator**: Evaluate mathematical expressions
- **JSON Processor**: Validate and transform JSON
- **Memory System**: Store and retrieve agent memory
- **Web Search**: (Mock implementation, ready for integration)

### ⚡ FastAPI Integration
- RESTful API for agent execution
- Real-time document ingestion
- Tool execution endpoints
- Health monitoring and detailed diagnostics

### 🐳 Docker Support
- Production-ready containerization
- Docker Compose for multi-service setup
- Volume management for persistence

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI App                           │
│  (Agent Execution, RAG, Tools, Health Monitoring)           │
└────────────────┬────────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌──────────┐
│  Agent  │ │   RAG   │ │   MCP    │
│ Engine  │ │Pipeline │ │  Server  │
└─┬───────┘ └────┬────┘ └────┬─────┘
  │              │            │
  ├──────────────┴────────────┤
  │                           │
  ▼                           ▼
┌──────────────┐      ┌──────────────┐
│ LLM Provider │      │ Vector Store │
│ (OpenAI...)  │      │ (Chromadb..) │
└──────────────┘      └──────────────┘
```

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional)
- OpenAI API key

### Installation

1. **Clone and setup**
```bash
cd /workspaces/Agentic_Exp
cp .env.example .env
# Edit .env with your OpenAI API key
```

2. **Install dependencies**
```bash
pip install -e .
```

3. **Configure environment**
```bash
export OPENAI_API_KEY="sk_..."
export LLM_MODEL="gpt-4-turbo-preview"
```

### Running Locally

**Start the API server:**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Access the API:**
- Main endpoint: http://localhost:8000/
- API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Running with Docker

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

## Usage Examples

### 1. Execute Agent with Self-Correction

```python
import asyncio
from src.agent import create_agent
from src.tools import create_tools

async def main():
    tools = create_tools()
    agent = create_agent(name="MyAgent", tools=tools)
    
    response = await agent.run(
        task="Solve this problem: What is 25% of 400?",
        max_iterations=5
    )
    
    print(f"Answer: {response.answer}")
    print(f"Steps: {len(response.steps)}")
    print(f"Corrections: {response.corrections_applied}")

asyncio.run(main())
```

### 2. Query RAG Knowledge Base

```python
from src.rag import get_rag_pipeline
import asyncio

async def main():
    rag = get_rag_pipeline()
    
    # Add documents
    rag.add_text("Python is great for data science")
    
    # Retrieve
    docs = await rag.retrieve("data science languages")
    print(docs)

asyncio.run(main())
```

### 3. Use MCP Tools

```python
from src.mcp import get_mcp_server
import asyncio

async def main():
    mcp = get_mcp_server()
    
    # Register custom tool
    async def my_tool(input: str) -> str:
        return f"Processed: {input}"
    
    mcp.register_tool(
        name="my_tool",
        description="My custom tool",
        handler=my_tool,
        input_schema={"input": "str"}
    )
    
    # Execute
    result = await mcp.execute_tool("my_tool", {"input": "hello"})
    print(result)

asyncio.run(main())
```

### 4. REST API Calls

**Execute Agent:**
```bash
curl -X POST http://localhost:8000/agent/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task": "What is machine learning?",
    "max_iterations": 5
  }'
```

**Add Documents to RAG:**
```bash
curl -X POST http://localhost:8000/rag/add-documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Machine learning enables systems to learn from data",
    "source": "docs"
  }'
```

**Retrieve Documents:**
```bash
curl -X POST http://localhost:8000/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how does machine learning work?",
    "top_k": 5
  }'
```

**List Tools:**
```bash
curl http://localhost:8000/mcp/tools
```

## Configuration

Edit `.env` or set environment variables:

```bash
# LLM
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.7

# RAG
RAG_VECTOR_STORE=chromadb
RAG_CHUNK_SIZE=1000
RAG_MAX_RETRIEVED_DOCS=5

# Agent
AGENT_MAX_ITERATIONS=10
AGENT_ENABLE_SELF_CORRECTION=true

# API
API_HOST=0.0.0.0
API_PORT=8000
```

## Project Structure

```
agentic-framework/
├── src/
│   ├── agent/           # Self-correcting agent engine
│   ├── rag/            # RAG pipeline
│   ├── mcp/            # MCP server
│   ├── tools/          # Built-in tools
│   ├── api/            # FastAPI application
│   └── utils.py        # Utility functions
├── config/
│   └── settings.py     # Configuration management
├── examples/           # Usage examples
├── tests/             # Test suite
├── Dockerfile         # Docker image
├── docker-compose.yml # Multi-service setup
├── pyproject.toml    # Dependencies
└── README.md
```

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Health check |
| GET | `/health/detailed` | Detailed health info |
| POST | `/agent/execute` | Run agent on task |
| POST | `/rag/add-documents` | Add documents to knowledge base |
| POST | `/rag/retrieve` | Query knowledge base |
| POST | `/tools/execute` | Execute a tool |
| GET | `/mcp/tools` | List available tools |
| GET | `/mcp/resources` | List available resources |

## Development

### Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/

# Lint
ruff check src/

# Type checking
mypy src/
```

### Running Examples

```bash
python examples/basic_agent.py
python examples/rag_demo.py
python examples/mcp_demo.py
```

## Advanced Features

### Custom Tools

Register custom tools with the MCP server:

```python
from src.mcp import get_mcp_server

mcp = get_mcp_server()

async def my_custom_tool(query: str) -> str:
    # Your implementation
    return f"Result for {query}"

mcp.register_tool(
    name="my_tool",
    description="My custom tool",
    handler=my_custom_tool,
    input_schema={"query": "str"}
)
```

### Custom RAG Sources

Add documents from various sources:

```python
rag = get_rag_pipeline()

# From text
rag.add_text("Some text content")

# From directory
rag.add_documents_from_directory("./documents", "**/*.pdf")
```

### Different Vector Stores

Configure in `.env`:

```bash
# ChromaDB (default, local)
RAG_VECTOR_STORE=chromadb

# Pinecone (cloud)
RAG_VECTOR_STORE=pinecone
PINECONE_API_KEY=your_key
```

## Performance Considerations

- **Agent Iterations**: Default is 10, adjust via `AGENT_MAX_ITERATIONS`
- **Chunk Size**: Larger chunks (1000+) for better context, smaller (500) for precision
- **Embedding Model**: `text-embedding-3-small` (cheaper), `text-embedding-3-large` (better)
- **Temperature**: Lower (0.3) for deterministic, higher (0.9) for creative responses

## Troubleshooting

### API won't start
- Check OPENAI_API_KEY is set
- Verify port 8000 is available
- Check logs: `tail -f logs/agentic_framework.log`

### RAG not retrieving documents
- Ensure documents were added: `GET /mcp/tools`
- Check embeddings are working
- Verify vector_store path exists

### Agent not improving
- Lower `AGENT_MAX_ITERATIONS` first
- Enable more verbose logging
- Check reflection prompt is appropriate

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes and test
4. Submit pull request

## Roadmap

- [ ] LLaMA/Ollama local LLM support
- [ ] Multi-agent coordination
- [ ] Advanced prompt caching
- [ ] Real-time streaming responses
- [ ] Web UI dashboard
- [ ] Graph-based reasoning
- [ ] Tool discovery and learning

## License

MIT License - see LICENSE file

## Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check examples in `/examples`
- Review tests in `/tests`
- Check logs in `logs/agentic_framework.log`

## References

- [LangChain Documentation](https://docs.langchain.com)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [OpenAI API](https://platform.openai.com/docs)
- [ChromaDB](https://docs.trychroma.com)