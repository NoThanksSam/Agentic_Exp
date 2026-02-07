"""Example usage of the agentic framework."""

import asyncio
from src.agent import create_agent
from src.rag import get_rag_pipeline
from src.mcp import get_mcp_server
from src.tools import create_tools
from config.settings import settings


async def main():
    """Example: Run an agent with RAG and MCP."""
    
    # Initialize components
    print("Initializing framework components...")
    rag_pipeline = get_rag_pipeline()
    mcp_server = get_mcp_server()
    
    # Add sample documents to RAG
    sample_text = """
    Python is a high-level programming language known for its simplicity and readability.
    It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    Python is widely used in data science, machine learning, web development, and automation.
    """
    
    print("Adding documents to RAG...")
    rag_pipeline.add_text(sample_text, metadata={"source": "example_docs"})
    
    # Create and configure agent
    print("Creating agent...")
    tools = create_tools()
    agent = create_agent(
        name="DemoAgent",
        tools=tools,
        model_name=settings.llm.model,
        temperature=settings.llm.temperature,
    )
    
    # Example task
    task = "Tell me about Python and its uses"
    context = "Use the retrieved knowledge base to answer."
    
    print(f"\nExecuting task: {task}")
    print("-" * 50)
    
    # Run agent
    response = await agent.run(
        task=task,
        context=context,
        max_iterations=3,
    )
    
    # Display results
    print(f"\nAgent Response:")
    print(f"Answer: {response.answer}")
    print(f"\nReasoning: {response.reasoning}")
    print(f"\nExecution Metrics:")
    print(f"  - Steps: {len(response.steps)}")
    print(f"  - Corrections Applied: {response.corrections_applied}")
    print(f"  - Execution Time: {response.execution_time_seconds:.2f}s")
    print(f"  - Success: {response.success}")


if __name__ == "__main__":
    asyncio.run(main())
