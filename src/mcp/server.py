"""Model Context Protocol (MCP) Server implementation."""

from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum
from loguru import logger
import json


class ToolInputSchema(BaseModel):
    """Tool input schema for MCP."""
    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)


class Tool(BaseModel):
    """MCP Tool definition."""
    name: str
    description: str
    input_schema: ToolInputSchema = Field(default_factory=ToolInputSchema)
    handler: Optional[Callable] = None
    
    class Config:
        arbitrary_types_allowed = True


class ResourceType(str, Enum):
    """Resource types for MCP."""
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    IMAGE = "image"


class Resource(BaseModel):
    """MCP Resource definition."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: str = "text/plain"
    data: Optional[str] = None


class PromptTemplate(BaseModel):
    """MCP Prompt template."""
    name: str
    description: str
    template: str
    arguments: List[str] = Field(default_factory=list)


class MCPServer:
    """Model Context Protocol Server for tool and resource management."""
    
    def __init__(self, name: str = "Agentic Framework MCP Server"):
        """Initialize MCP server."""
        self.name = name
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Resource] = {}
        self.prompts: Dict[str, PromptTemplate] = {}
        logger.info(f"Initialized MCP Server: {name}")
    
    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a tool with the MCP server."""
        schema = ToolInputSchema(
            properties=input_schema or {},
            required=list(input_schema.keys()) if input_schema else []
        )
        
        tool = Tool(
            name=name,
            description=description,
            input_schema=schema,
            handler=handler
        )
        
        self.tools[name] = tool
        logger.info(f"Registered tool: {name}")
    
    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str = "text/plain",
        data: Optional[str] = None,
    ) -> None:
        """Register a resource with the MCP server."""
        resource = Resource(
            uri=uri,
            name=name,
            description=description,
            mime_type=mime_type,
            data=data
        )
        
        self.resources[uri] = resource
        logger.info(f"Registered resource: {name} ({uri})")
    
    def register_prompt(
        self,
        name: str,
        description: str,
        template: str,
        arguments: Optional[List[str]] = None,
    ) -> None:
        """Register a prompt template with the MCP server."""
        prompt = PromptTemplate(
            name=name,
            description=description,
            template=template,
            arguments=arguments or []
        )
        
        self.prompts[name] = prompt
        logger.info(f"Registered prompt: {name}")
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        tool = self.tools[tool_name]
        if not tool.handler:
            raise ValueError(f"Tool has no handler: {tool_name}")
        
        try:
            logger.info(f"Executing tool: {tool_name} with args: {arguments}")
            
            # Check if handler is async
            import inspect
            if inspect.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)
            
            logger.info(f"Tool {tool_name} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            raise
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a registered tool."""
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all registered tools as dictionaries."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema.model_dump()
            }
            for tool in self.tools.values()
        ]
    
    def get_resource(self, uri: str) -> Optional[Resource]:
        """Get a registered resource."""
        return self.resources.get(uri)
    
    def get_all_resources(self) -> List[Dict[str, Any]]:
        """Get all registered resources as dictionaries."""
        return [
            {
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mime_type
            }
            for resource in self.resources.values()
        ]
    
    def get_prompt(self, prompt_name: str) -> Optional[PromptTemplate]:
        """Get a registered prompt template."""
        return self.prompts.get(prompt_name)
    
    def render_prompt(self, prompt_name: str, **kwargs) -> str:
        """Render a prompt template with arguments."""
        prompt = self.prompts.get(prompt_name)
        if not prompt:
            raise ValueError(f"Prompt not found: {prompt_name}")
        
        return prompt.template.format(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export server configuration as dictionary."""
        return {
            "name": self.name,
            "tools": self.get_all_tools(),
            "resources": self.get_all_resources(),
            "prompts": [
                {
                    "name": p.name,
                    "description": p.description,
                    "arguments": p.arguments
                }
                for p in self.prompts.values()
            ]
        }


# Global MCP server instance
_mcp_server: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """Get or create the MCP server."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
    return _mcp_server
