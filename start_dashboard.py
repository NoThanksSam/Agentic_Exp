#!/usr/bin/env python3
"""Quick start for Agentic Framework with dashboard."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging
from src.agent import create_agent
from src.rag import get_rag_pipeline
from src.mcp import get_mcp_server
from src.monitoring import get_metrics_collector, get_hallucination_detector, get_audit_logger
from src.tools import create_tools, register_tools_with_mcp
from config.settings import settings


def print_header(text: str, width: int = 60):
    """Print a formatted header."""
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}\n")


async def main():
    """Main entry point."""
    
    print_header("🚀 Agentic Framework Dashboard Setup")
    
    # Setup logging
    setup_logging(level=settings.log_level)
    
    # Check API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set")
        print("   Set your OpenAI API key before running the dashboard:")
        print("   export OPENAI_API_KEY='sk_...'")
        print("\nTo continue without key, press Enter (features will be limited)")
        input()
    
    print("✅ Framework initialized!")
    print(f"   - LLM: {settings.llm.provider} ({settings.llm.model})")
    print(f"   - RAG: {settings.rag.vector_store}")
    print(f"   - Max iterations: {settings.agent.max_iterations}")
    
    # Initialize components
    print_header("Component Initialization")
    
    print("📚 RAG Pipeline...")
    rag_pipeline = get_rag_pipeline()
    print("   ✓ Ready")
    
    print("🔌 MCP Server...")
    mcp_server = get_mcp_server()
    print("   ✓ Ready")
    
    print("📊 Metrics Collector...")
    metrics = get_metrics_collector()
    print("   ✓ Ready")
    
    print("🔍 Hallucination Detector...")
    hallu_detector = get_hallucination_detector()
    print("   ✓ Ready")
    
    print("📋 Audit Logger...")
    audit = get_audit_logger()
    print("   ✓ Ready")
    
    print("🛠️  Tools...")
    tools = create_tools()
    register_tools_with_mcp(mcp_server)
    print(f"   ✓ {len(mcp_server.tools)} tools registered")
    
    print("🤖 Agent Engine...")
    agent = create_agent(
        name="DashboardAgent",
        tools=tools,
        model_name=settings.llm.model,
        temperature=settings.llm.temperature,
    )
    print("   ✓ Ready")
    
    # Demo: Add sample documents
    print_header("Loading Sample Knowledge Base")
    
    sample_docs = {
        "ai_basics": """
Artificial Intelligence (AI) is the branch of computer science dealing with the simulation of 
intelligent behavior in computers. AI systems are designed to perform tasks that typically require 
human intelligence, including visual perception, speech recognition, decision-making, and language 
translation. Key AI techniques include machine learning, deep learning, and natural language processing.
        """,
        "ml_guide": """
Machine Learning is a subset of artificial intelligence that focuses on the ability of computer 
systems to learn from and make predictions based on data, without being explicitly programmed. 
Machine learning algorithms can be classified into three main types: supervised learning (with labeled data), 
unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through 
interaction with the environment).
        """,
        "neural_networks": """
Neural networks are computing systems inspired by biological neural networks found in animal brains. 
They consist of interconnected nodes (neurons) organized in layers that process information together. 
Deep neural networks with multiple hidden layers (deep learning) have revolutionized AI, enabling 
breakthrough capabilities in image recognition, language understanding, and game playing.
        """,
    }
    
    for name, content in sample_docs.items():
        rag_pipeline.add_text(content, metadata={"source": name, "type": "sample"})
        print(f"   ✓ Loaded: {name}")
    
    print(f"   📊 Total documents in knowledge base: {len(rag_pipeline.documents)}")
    
    # Info for starting dashboard
    print_header("🎯 Ready to Use!")
    
    print("To start the API server with dashboard, run:")
    print("\n  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000\n")
    print("Then open in your browser:")
    print("  http://localhost:8000/dashboard/\n")
    
    print("API Documentation:")
    print("  http://localhost:8000/docs\n")
    
    # Quick demo
    print_header("Quick Demo: Chat with Agent")
    
    demo_queries = [
        "What is machine learning?",
        "How do neural networks work?",
    ]
    
    for query in demo_queries[:1]:  # Just run first one for demo
        print(f"\n📨 User: {query}")
        
        try:
            response = await agent.run(
                task=query,
                context=None,  # Will use RAG
                max_iterations=2,
            )
            
            print(f"🤖 Agent: {response.answer[:200]}...")
            
            # Check for hallucinations
            hal_detector = get_hallucination_detector()
            hallucination = hal_detector.analyze(
                response=response.answer,
                supporting_documents=[doc.page_content for doc in rag_pipeline.documents.values()]
            )
            
            if hallucination.is_hallucinated:
                print(f"⚠️  Hallucination Risk: {hallucination.risk_level.value}")
                print(f"   Reasons: {', '.join(hallucination.reasons[:2])}")
            else:
                print(f"✅ Hallucination Check: Passed")
            
            # Show metrics
            metrics_obj = metrics.get_stats("DashboardAgent")
            print(f"\n📊 Metrics:")
            print(f"   Executions: {metrics_obj.total_executions}")
            print(f"   Success Rate: {metrics_obj.success_rate:.1f}%")
            print(f"   Avg Time: {metrics_obj.avg_execution_time:.2f}s")
            print(f"   Corrections: {metrics_obj.total_corrections}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    # Features overview
    print_header("Dashboard Features")
    
    features = [
        ("💬 Real-time Chat", "Talk to agents in plain English"),
        ("📊 Live Metrics", "Monitor execution time, success rate, efficiency"),
        ("🔍 Hallucination Detection", "Identify suspicious/unsupported claims"),
        ("📋 Audit Trail", "Complete log of all actions and decisions"),
        ("📈 Performance Tracking", "Self-corrections, tools used, confidence"),
        ("🌐 WebSocket Support", "Real-time updates without polling"),
        ("📤 Export Capabilities", "Download audit logs as JSON"),
        ("🔧 RESTful API", "Full API access for programmatic usage"),
    ]
    
    print("Available Features:\n")
    for i, (feature, description) in enumerate(features, 1):
        print(f"  {i}. {feature:25} - {description}")
    
    # Next steps
    print_header("Next Steps")
    
    print("\n1. Start the API Server")
    print("   $ uvicorn src.api.main:app --reload\n")
    
    print("2. Open Dashboard")
    print("   Visit: http://localhost:8000/dashboard/\n")
    
    print("3. Start Chatting")
    print("   Type a question in plain English\n")
    
    print("4. Monitor Performance")
    print("   Watch metrics update in real-time\n")
    
    print("5. Review Audit Trail")
    print("   See all actions in the audit log\n")
    
    print("💡 Pro Tips:")
    print("   • Use multi-turn conversations for context")
    print("   • Watch the audit log for hallucination warnings")
    print("   • Add more documents for better RAG accuracy")
    print("   • Export audit logs for compliance\n")
    
    print_header("Framework Ready! 🎉")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
