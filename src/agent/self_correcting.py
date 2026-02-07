"""Self-Correcting Agent Framework."""

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
import json
from loguru import logger
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, StringPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential


class AgentState(str, Enum):
    """Agent execution state."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRECTING = "correcting"


class ToolCall(BaseModel):
    """A tool call within an agent step."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    executed_at: datetime = Field(default_factory=datetime.utcnow)


class AgentStep(BaseModel):
    """A single step in agent reasoning."""
    step_number: int
    thought: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    observation: Optional[str] = None
    confidence: float = 0.5
    is_correction: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentResponse(BaseModel):
    """Final response from an agent."""
    answer: str
    reasoning: str
    steps: List[AgentStep] = Field(default_factory=list)
    success: bool = True
    corrections_applied: int = 0
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReflectionResult(BaseModel):
    """Result of agent reflection/self-correction."""
    needs_correction: bool
    issues_found: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    feedback: str = ""


class SelfCorrectingAgent:
    """Agent with built-in self-correction and reflection capabilities."""
    
    def __init__(
        self,
        name: str = "SelfCorrectingAgent",
        tools: Optional[List[Tool]] = None,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
    ):
        """Initialize the self-correcting agent."""
        self.name = name
        self.tools = tools or []
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )
        self.steps: List[AgentStep] = []
        self.state = AgentState.PENDING
        self.corrections_count = 0
        
        logger.info(f"Initialized {name} with {len(self.tools)} tools")
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.tools.append(tool)
        logger.info(f"Added tool: {tool.name}")
    
    async def think(self, task: str, context: str = "") -> str:
        """Generate reasoning for the task."""
        prompt = ChatPromptTemplate.from_template(
            """You are a reasoning agent. Analyze the task and generate clear reasoning.
            
Task: {task}
{context_section}

Provide your thought process:"""
        )
        
        context_section = f"Context:\n{context}" if context else ""
        
        chain = prompt | self.model
        response = await chain.ainvoke({
            "task": task,
            "context_section": context_section
        })
        
        return response.content
    
    async def reflect(self, response: str, task: str, steps: List[AgentStep]) -> ReflectionResult:
        """Self-reflect on the generated response."""
        if not response:
            return ReflectionResult(
                needs_correction=True,
                issues_found=["Empty response"],
                feedback="Response is empty"
            )
        
        prompt = ChatPromptTemplate.from_template(
            """Evaluate the following response to the task. Identify any issues or areas for improvement.

Task: {task}
Response: {response}

Previous steps taken: {steps}

Provide evaluation in JSON format:
{{
    "needs_correction": boolean,
    "issues_found": [list of issues],
    "suggestions": [list of suggestions],
    "feedback": "overall feedback"
}}"""
        )
        
        steps_text = "\n".join([
            f"Step {s.step_number}: {s.thought[:100]}..."
            for s in steps
        ])
        
        chain = prompt | self.model
        response_obj = await chain.ainvoke({
            "task": task,
            "response": response[:1000],
            "steps": steps_text
        })
        
        try:
            import json
            reflection_dict = json.loads(response_obj.content)
            return ReflectionResult(**reflection_dict)
        except Exception as e:
            logger.warning(f"Failed to parse reflection: {e}")
            return ReflectionResult(
                needs_correction=False,
                feedback=response_obj.content
            )
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """Execute a tool with error handling."""
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            error_msg = f"Tool not found: {tool_name}"
            logger.error(error_msg)
            return None, error_msg
        
        try:
            logger.info(f"Executing tool: {tool_name}")
            result = await tool.func(**arguments) if hasattr(tool, 'func') else tool(**arguments)
            return result, None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool execution error: {error_msg}")
            return None, error_msg
    
    async def run(
        self,
        task: str,
        context: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> AgentResponse:
        """Run the agent on a task with self-correction."""
        import time
        start_time = time.time()
        
        self.state = AgentState.EXECUTING
        self.steps = []
        self.corrections_count = 0
        
        max_iterations = max_iterations or 10
        logger.info(f"Starting agent execution: {task}")
        
        # Initial thinking
        thought = await self.think(task, context or "")
        
        step = AgentStep(
            step_number=1,
            thought=thought,
        )
        self.steps.append(step)
        
        # Iterative refinement loop
        answer = thought
        for iteration in range(max_iterations):
            # Reflect on current answer
            if iteration > 0 and self.steps:  # Skip reflection on first step
                reflection = await self.reflect(answer, task, self.steps)
                
                if reflection.needs_correction:
                    self.state = AgentState.CORRECTING
                    self.corrections_count += 1
                    
                    # Generate correction
                    correction_prompt = ChatPromptTemplate.from_template(
                        """The previous response had issues. Please improve it.
                        
Original response: {response}
Issues: {issues}
Suggestions: {suggestions}

Generate an improved response:"""
                    )
                    
                    chain = correction_prompt | self.model
                    correction_resp = await chain.ainvoke({
                        "response": answer[:500],
                        "issues": ", ".join(reflection.issues_found),
                        "suggestions": ", ".join(reflection.suggestions)
                    })
                    
                    answer = correction_resp.content
                    
                    correction_step = AgentStep(
                        step_number=len(self.steps) + 1,
                        thought=f"Correction applied: {reflection.feedback[:100]}",
                        is_correction=True,
                        observation=answer[:200]
                    )
                    self.steps.append(correction_step)
                    logger.info(f"Applied correction {self.corrections_count}")
                else:
                    logger.info("Reflection passed - no correction needed")
                    break
            
            if iteration >= max_iterations - 1:
                logger.warning(f"Max iterations ({max_iterations}) reached")
                break
        
        self.state = AgentState.COMPLETED
        execution_time = time.time() - start_time
        
        logger.info(f"Agent execution completed in {execution_time:.2f}s")
        
        return AgentResponse(
            answer=answer,
            reasoning=self.steps[0].thought if self.steps else "",
            steps=self.steps,
            success=True,
            corrections_applied=self.corrections_count,
            execution_time_seconds=execution_time,
            metadata={
                "model": self.model.model_name,
                "tools_used": len([s for step in self.steps for s in step.tool_calls]),
                "iterations": len(self.steps),
            }
        )


def create_agent(
    name: str = "DefaultAgent",
    tools: Optional[List[Tool]] = None,
    **kwargs
) -> SelfCorrectingAgent:
    """Factory function to create a self-correcting agent."""
    return SelfCorrectingAgent(name=name, tools=tools or [], **kwargs)
