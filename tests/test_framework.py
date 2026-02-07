"""Tests for the agentic framework."""

import pytest
import asyncio
from src.agent import create_agent, AgentResponse
from src.rag import RAGPipeline
from src.mcp import get_mcp_server
from src.tools import create_tools


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def agent():
    """Create a test agent."""
    tools = create_tools()
    return create_agent(
        name="TestAgent",
        tools=tools,
        model_name="gpt-3.5-turbo",
        temperature=0.5,
    )


@pytest.fixture
def rag_pipeline():
    """Create a test RAG pipeline."""
    return RAGPipeline()


@pytest.fixture
def mcp_server():
    """Get MCP server."""
    return get_mcp_server()


class TestAgent:
    """Test agent functionality."""
    
    @pytest.mark.asyncio
    async def test_agent_creation(self, agent):
        """Test agent can be created."""
        assert agent is not None
        assert agent.name == "TestAgent"
        assert len(agent.tools) > 0
    
    @pytest.mark.asyncio
    async def test_agent_think(self, agent):
        """Test agent can think about a task."""
        thought = await agent.think("What is 2+2?")
        assert thought is not None
        assert len(thought) > 0
    
    @pytest.mark.asyncio
    async def test_agent_reflect(self, agent):
        """Test agent can reflect on responses."""
        response = "The answer is 4"
        task = "What is 2+2?"
        steps = []
        
        reflection = await agent.reflect(response, task, steps)
        assert reflection is not None
        assert hasattr(reflection, 'needs_correction')


class TestRAGPipeline:
    """Test RAG pipeline."""
    
    def test_rag_pipeline_creation(self, rag_pipeline):
        """Test RAG pipeline can be created."""
        assert rag_pipeline is not None
    
    def test_add_text_to_rag(self, rag_pipeline):
        """Test adding text to RAG."""
        text = "The quick brown fox jumps over the lazy dog."
        rag_pipeline.add_text(text)
        
        assert len(rag_pipeline.documents) > 0
    
    @pytest.mark.asyncio
    async def test_retrieve_from_rag(self, rag_pipeline):
        """Test retrieving documents from RAG."""
        text = "Python is a programming language. Java is also a language."
        rag_pipeline.add_text(text)
        
        documents = await rag_pipeline.retrieve("programming")
        # Check that something was returned (might be empty due to mock)
        assert isinstance(documents, list)


class TestMCPServer:
    """Test MCP server."""
    
    def test_mcp_server_creation(self, mcp_server):
        """Test MCP server can be created."""
        assert mcp_server is not None
    
    def test_register_tool_with_mcp(self, mcp_server):
        """Test registering a tool with MCP."""
        async def test_tool(arg: str) -> str:
            return f"Test: {arg}"
        
        mcp_server.register_tool(
            name="test_tool",
            description="A test tool",
            handler=test_tool,
            input_schema={"arg": "str"}
        )
        
        assert "test_tool" in mcp_server.tools
    
    @pytest.mark.asyncio
    async def test_execute_mcp_tool(self, mcp_server):
        """Test executing a tool via MCP."""
        async def test_tool(arg: str) -> str:
            return f"Result: {arg}"
        
        mcp_server.register_tool(
            name="test_tool",
            description="A test tool",
            handler=test_tool,
            input_schema={"arg": "str"}
        )
        
        result = await mcp_server.execute_tool("test_tool", {"arg": "hello"})
        assert result == "Result: hello"
    
    def test_register_resource_with_mcp(self, mcp_server):
        """Test registering a resource with MCP."""
        mcp_server.register_resource(
            uri="memory://test",
            name="Test Resource",
            description="A test resource",
            data="Test data"
        )
        
        assert "memory://test" in mcp_server.resources


class TestTools:
    """Test built-in tools."""
    
    def test_calculator_tool(self):
        """Test calculator tool."""
        from src.tools import calculator_tool
        
        result = calculator_tool("2 + 2")
        assert "4" in result
    
    def test_json_processor_tool(self):
        """Test JSON processor tool."""
        from src.tools import json_processor_tool
        
        valid_json = '{"key": "value"}'
        result = json_processor_tool(valid_json, "validate")
        assert "valid" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
