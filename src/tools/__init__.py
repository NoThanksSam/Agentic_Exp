"""Built-in tools for the agentic framework."""

from typing import Any, Dict, List
from langchain.agents import Tool
from loguru import logger
import json
from datetime import datetime
from src.rag import get_rag_pipeline
from src.mcp import get_mcp_server


# RAG Tool
async def retrieve_documents_tool(query: str, top_k: int = 5) -> str:
    """Retrieve relevant documents from the knowledge base."""
    try:
        rag_pipeline = get_rag_pipeline()
        documents = await rag_pipeline.retrieve(query, k=top_k)
        
        if not documents:
            return "No relevant documents found."
        
        context = rag_pipeline.get_context_from_documents(documents)
        return context[:2000]  # Return limited context
    except Exception as e:
        logger.error(f"Error in retrieve_documents_tool: {e}")
        return f"Error retrieving documents: {str(e)}"


# Web Search Tool (mock)
def web_search_tool(query: str, num_results: int = 5) -> str:
    """Search the web for information."""
    logger.info(f"Web search for: {query}")
    # This is a mock implementation
    return f"Web search results for '{query}' (mocked)"


# Calculator Tool
def calculator_tool(expression: str) -> str:
    """Evaluate mathematical expressions safely."""
    try:
        import math
        # Safe evaluation with limited namespace
        allow_list = {
            'pi': math.pi,
            'e': math.e,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'exp': math.exp,
        }
        result = eval(expression, {"__builtins__": {}}, allow_list)
        return f"Result: {result}"
    except Exception as e:
        logger.error(f"Calculator error: {e}")
        return f"Error evaluating expression: {str(e)}"


# JSON Processing Tool
def json_processor_tool(json_string: str, operation: str = "validate") -> str:
    """Process and validate JSON data."""
    try:
        data = json.loads(json_string)
        
        if operation == "validate":
            return "JSON is valid"
        elif operation == "pretty":
            return json.dumps(data, indent=2)
        elif operation == "minify":
            return json.dumps(data, separators=(',', ':'))
        else:
            return f"Unknown operation: {operation}"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {str(e)}"


# Memory Tool
memory_store = {}

def memory_store_tool(key: str, value: str) -> str:
    """Store information in agent memory."""
    memory_store[key] = {
        "value": value,
        "timestamp": datetime.utcnow().isoformat()
    }
    logger.info(f"Stored in memory: {key}")
    return f"Stored '{key}' in memory"


def memory_retrieve_tool(key: str) -> str:
    """Retrieve information from agent memory."""
    if key in memory_store:
        return memory_store[key]["value"]
    else:
        return f"Key not found in memory: {key}"


def memory_list_tool() -> str:
    """List all keys in agent memory."""
    keys = list(memory_store.keys())
    return f"Memory keys: {', '.join(keys)}" if keys else "Memory is empty"


# Tool registration function
def create_tools() -> List[Tool]:
    """Create and return all available tools."""
    tools = [
        Tool(
            name="retrieve_documents",
            func=retrieve_documents_tool,
            description="Retrieve relevant documents from the knowledge base. Use this to find information about topics.",
            args_schema=None,
        ),
        Tool(
            name="web_search",
            func=web_search_tool,
            description="Search the web for information. Use when you need current information or facts.",
            args_schema=None,
        ),
        Tool(
            name="calculator",
            func=calculator_tool,
            description="Evaluate mathematical expressions. Use for calculations and math.",
            args_schema=None,
        ),
        Tool(
            name="json_processor",
            func=json_processor_tool,
            description="Process and validate JSON data. Operations: validate, pretty, minify.",
            args_schema=None,
        ),
        Tool(
            name="memory_store",
            func=memory_store_tool,
            description="Store information in agent memory for later retrieval.",
            args_schema=None,
        ),
        Tool(
            name="memory_retrieve",
            func=memory_retrieve_tool,
            description="Retrieve stored information from agent memory.",
            args_schema=None,
        ),
        Tool(
            name="memory_list",
            func=memory_list_tool,
            description="List all keys stored in agent memory.",
            args_schema=None,
        ),
    ]
    
    logger.info(f"Created {len(tools)} tools")
    return tools


# Register tools with MCP server
def register_tools_with_mcp(mcp_server) -> None:
    """Register all tools with the MCP server."""
    
    mcp_server.register_tool(
        name="retrieve_documents",
        description="Retrieve relevant documents from knowledge base",
        handler=retrieve_documents_tool,
        input_schema={"query": "str", "top_k": "int"}
    )
    
    mcp_server.register_tool(
        name="calculator",
        description="Evaluate mathematical expressions",
        handler=calculator_tool,
        input_schema={"expression": "str"}
    )
    
    mcp_server.register_tool(
        name="memory_store",
        description="Store data in agent memory",
        handler=memory_store_tool,
        input_schema={"key": "str", "value": "str"}
    )
    
    mcp_server.register_tool(
        name="memory_retrieve",
        description="Retrieve data from agent memory",
        handler=memory_retrieve_tool,
        input_schema={"key": "str"}
    )
    
    logger.info("Registered tools with MCP server")
