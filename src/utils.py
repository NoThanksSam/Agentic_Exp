"""Utility functions for the framework."""

import json
from typing import Any, Dict
from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
    )
    logger.add(
        "logs/agentic_framework.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=level,
        rotation="500 MB",
    )


def serialize_tool_output(output: Any) -> str:
    """Serialize tool output to JSON string."""
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to serialize output: {e}")
        return str(output)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text safely."""
    try:
        # Try to find JSON block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            json_str = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            json_str = text[start:end].strip()
        else:
            json_str = text
        
        return json.loads(json_str)
    except Exception as e:
        logger.warning(f"Failed to extract JSON: {e}")
        return {}


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max length."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text
