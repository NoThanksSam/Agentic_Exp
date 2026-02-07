"""Quick start script to initialize and run the agentic framework."""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging
from src.agent import create_agent
from src.rag import get_rag_pipeline
from src.mcp import get_mcp_server
from src.tools import create_tools, register_tools_with_mcp
from config.settings import settings


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


async def main():
    """Main entry point."""
    
    print_header("Agentic Framework - Quick Start")
    
    # Setup logging
    setup_logging(level=settings.log_level)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set")
        print("   Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='sk_...'")
        print("\n   Or create a .env file with the key")
        return
    
    print("✅ Configuration loaded")
    print(f"   - LLM: {settings.llm.provider} ({settings.llm.model})")
    print(f"   - RAG: {settings.rag.vector_store}")
    print(f"   - Max Agent Iterations: {settings.agent.max_iterations}")
    
    # Initialize components
    print_header("Initializing Components")
    
    print("📚 RAG Pipeline...")
    rag_pipeline = get_rag_pipeline()
    
    print("🔌 MCP Server...")
    mcp_server = get_mcp_server()
    
    print("🛠️  Tools...")
    tools = create_tools()
    register_tools_with_mcp(mcp_server)
    print(f"   Registered {len(mcp_server.tools)} tools")
    
    print("🤖 Agent Engine...")
    agent = create_agent(
        name="QuickStartAgent",
        tools=tools,
        model_name=settings.llm.model,
        temperature=settings.llm.temperature,
    )
    
    # Demo: Add sample documents
    print_header("Adding Sample Documents")
    
    sample_docs = {
        "ai_basics": """
        Artificial Intelligence (AI) is the ability of computer systems to perform 
        tasks that typically require human intelligence. This includes learning 
        from experience, recognizing patterns, understanding language, and making 
        decisions. AI applications range from virtual assistants to automation 
        systems and predictive analytics.
        """,
        "ml_intro": """
        Machine Learning is a subset of AI that enables systems to learn and improve
        from experience without being explicitly programmed. Machine learning 
        algorithms identify patterns in data and use these patterns to predict 
        outcomes on new, unseen data. Common ML techniques include supervised learning, 
        unsupervised learning, and reinforcement learning.
        """,
    }
    
    for name, content in sample_docs.items():
        rag_pipeline.add_text(content, metadata={"source": name})
        print(f"   ✓ Added: {name}")
    
    # Demo: Run agent
    print_header("Running Agent Demo")
    
    demo_tasks = [
        "What is artificial intelligence and what are its main applications?",
        "Explain the difference between AI and machine learning",
    ]
    
    for task in demo_tasks:
        print(f"📋 Task: {task}\n")
        
        response = await agent.run(
            task=task,
            context="Use the knowledge base to provide accurate answers.",
            max_iterations=3,
        )
        
        print(f"✨ Answer:\n{response.answer}\n")
        print(f"📊 Metrics:")
        print(f"   - Steps: {len(response.steps)}")
        print(f"   - Self-corrections: {response.corrections_applied}")
        print(f"   - Execution time: {response.execution_time_seconds:.2f}s\n")
    
    # API info
    print_header("Next Steps")
    
    print("To start the API server, run:")
    print("  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")
    print("\nTo use Docker, run:")
    print("  docker-compose up -d")
    print("\nAPI will be available at:")
    print("  - Main: http://localhost:8000/")
    print("  - Docs: http://localhost:8000/docs")
    print("\nTo run examples:")
    print("  python examples/basic_agent.py")
    print("  python examples/rag_demo.py")
    print("  python examples/mcp_demo.py")
    
    print_header("Framework Ready! 🚀")


if __name__ == "__main__":
    asyncio.run(main())
