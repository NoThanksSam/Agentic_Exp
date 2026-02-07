"""Configuration management for the agentic framework."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, Literal
import os


class LLMSettings(BaseSettings):
    """LLM configuration."""
    provider: Literal["openai", "anthropic", "local"] = "openai"
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    
    class Config:
        env_prefix = "LLM_"


class RAGSettings(BaseSettings):
    """RAG configuration."""
    vector_store: Literal["chromadb", "pinecone", "weaviate"] = "chromadb"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_threshold: float = 0.5
    max_retrieved_docs: int = 5
    
    # Pinecone specific
    pinecone_api_key: Optional[str] = Field(default=None, validation_alias="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, validation_alias="PINECONE_ENV")
    pinecone_index: str = "agentic-framework"
    
    # ChromaDB specific
    chromadb_path: str = "./data/chromadb"
    
    class Config:
        env_prefix = "RAG_"


class AgentSettings(BaseSettings):
    """Agent configuration."""
    max_iterations: int = 10
    max_retries: int = 3
    enable_self_correction: bool = True
    enable_reflection: bool = True
    reflection_prompt_template: str = "Evaluate your previous response against the task requirements. What could be improved?"
    enable_tool_validation: bool = True
    timeout_seconds: int = 300
    
    class Config:
        env_prefix = "AGENT_"


class MCPSettings(BaseSettings):
    """Model Context Protocol configuration."""
    enabled: bool = True
    server_host: str = "0.0.0.0"
    server_port: int = 8001
    debug: bool = False
    
    class Config:
        env_prefix = "MCP_"


class APISettings(BaseSettings):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = True
    project_name: str = "Agentic Framework"
    
    class Config:
        env_prefix = "API_"


class AppSettings(BaseSettings):
    """Main application settings."""
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    
    llm: LLMSettings = LLMSettings()
    rag: RAGSettings = RAGSettings()
    agent: AgentSettings = AgentSettings()
    mcp: MCPSettings = MCPSettings()
    api: APISettings = APISettings()
    
    class Config:
        case_sensitive = False
        env_nested_delimiter = "__"


def get_settings() -> AppSettings:
    """Get application settings."""
    return AppSettings(
        llm=LLMSettings(),
        rag=RAGSettings(),
        agent=AgentSettings(),
        mcp=MCPSettings(),
        api=APISettings(),
    )


# Global settings instance
settings = get_settings()
