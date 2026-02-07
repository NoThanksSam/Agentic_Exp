"""Example: Using MCP server and tools programmatically."""

import asyncio
from src.mcp import get_mcp_server
from src.tools import create_tools, register_tools_with_mcp


async def main():
    """Example: Interact with MCP server."""
    
    print("Initializing MCP Server...")
    mcp_server = get_mcp_server()
    
    # Register tools
    print("Registering tools...")
    register_tools_with_mcp(mcp_server)
    
    # List available tools
    print("\nAvailable Tools:")
    print("-" * 50)
    for tool in mcp_server.get_all_tools():
        print(f"- {tool['name']}: {tool['description']}")
    
    # Execute a tool
    print("\n\nExecuting calculator tool...")
    result = await mcp_server.execute_tool("calculator", {"expression": "10 * 5 + 3"})
    print(f"Result: {result}")
    
    # Use memory tools
    print("\n\nUsing memory tools...")
    await mcp_server.execute_tool("memory_store", {
        "key": "company",
        "value": "Acme Corporation"
    })
    print("Stored in memory: company=Acme Corporation")
    
    retrieved = await mcp_server.execute_tool("memory_retrieve", {"key": "company"})
    print(f"Retrieved from memory: {retrieved}")
    
    # List all stored memory keys
    keys = await mcp_server.execute_tool("memory_list", {})
    print(f"Memory contents: {keys}")
    
    # Register custom prompt
    print("\n\nRegistering custom prompt...")
    mcp_server.register_prompt(
        name="problem_solver",
        description="Prompt to help solve problems",
        template="Please solve this problem step by step: {problem}",
        arguments=["problem"]
    )
    
    # Render prompt
    rendered = mcp_server.render_prompt(
        "problem_solver",
        problem="How to optimize a database query?"
    )
    print(f"Rendered prompt: {rendered}")


if __name__ == "__main__":
    asyncio.run(main())
