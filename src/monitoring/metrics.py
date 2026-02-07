"""Metrics tracking and monitoring system."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import time
from loguru import logger


class MetricType(str, Enum):
    """Types of metrics tracked."""
    EXECUTION_TIME = "execution_time"
    TOKENS_USED = "tokens_used"
    TOOLS_EXECUTED = "tools_executed"
    CORRECTIONS_APPLIED = "corrections_applied"
    ACCURACY = "accuracy"
    CONFIDENCE = "confidence"


@dataclass
class Metric:
    """A single metric data point."""
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_id: str = "default"
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float = 0.0
    avg_confidence: float = 0.0
    total_corrections: int = 0
    total_tools_used: int = 0
    success_rate: float = 0.0
    avg_corrections_per_execution: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """Collects and aggregates metrics from agent executions."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector."""
        self.max_history = max_history
        self.metrics: List[Metric] = []
        self.execution_times: Dict[str, List[float]] = {}
        self.confidence_scores: Dict[str, List[float]] = {}
        logger.info(f"Initialized MetricsCollector with max_history={max_history}")
    
    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        agent_id: str = "default",
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a metric."""
        metric = Metric(
            metric_type=metric_type,
            value=value,
            agent_id=agent_id,
            session_id=session_id,
            metadata=metadata or {},
        )
        
        self.metrics.append(metric)
        
        # Maintain history limit
        if len(self.metrics) > self.max_history:
            self.metrics.pop(0)
        
        # Update aggregate tracking
        if metric_type == MetricType.EXECUTION_TIME:
            self.execution_times.setdefault(agent_id, []).append(value)
        elif metric_type == MetricType.CONFIDENCE:
            self.confidence_scores.setdefault(agent_id, []).append(value)
        
        logger.debug(f"Recorded metric: {metric_type.value}={value}")
    
    def record_execution(
        self,
        agent_id: str,
        execution_time: float,
        success: bool,
        confidence: float = 0.5,
        corrections_applied: int = 0,
        tools_used: int = 0,
        session_id: str = "",
    ) -> None:
        """Record a complete agent execution."""
        self.record_metric(
            MetricType.EXECUTION_TIME,
            execution_time,
            agent_id=agent_id,
            session_id=session_id,
            metadata={"success": success},
        )
        
        self.record_metric(
            MetricType.CONFIDENCE,
            confidence,
            agent_id=agent_id,
            session_id=session_id,
        )
        
        self.record_metric(
            MetricType.CORRECTIONS_APPLIED,
            corrections_applied,
            agent_id=agent_id,
            session_id=session_id,
        )
        
        self.record_metric(
            MetricType.TOOLS_EXECUTED,
            tools_used,
            agent_id=agent_id,
            session_id=session_id,
        )
    
    def get_stats(self, agent_id: str = "default") -> PerformanceStats:
        """Get performance statistics for an agent."""
        agent_metrics = [m for m in self.metrics if m.agent_id == agent_id]
        
        if not agent_metrics:
            return PerformanceStats()
        
        exec_metrics = [m for m in agent_metrics if m.metric_type == MetricType.EXECUTION_TIME]
        conf_metrics = [m for m in agent_metrics if m.metric_type == MetricType.CONFIDENCE]
        corr_metrics = [m for m in agent_metrics if m.metric_type == MetricType.CORRECTIONS_APPLIED]
        tools_metrics = [m for m in agent_metrics if m.metric_type == MetricType.TOOLS_EXECUTED]
        
        successful = sum(1 for m in exec_metrics if m.metadata.get("success", True))
        total = len(exec_metrics)
        
        stats = PerformanceStats(
            total_executions=total,
            successful_executions=successful,
            failed_executions=total - successful,
            avg_execution_time=sum(m.value for m in exec_metrics) / len(exec_metrics) if exec_metrics else 0,
            avg_confidence=sum(m.value for m in conf_metrics) / len(conf_metrics) if conf_metrics else 0,
            total_corrections=sum(m.value for m in corr_metrics),
            total_tools_used=sum(m.value for m in tools_metrics),
            success_rate=(successful / total * 100) if total > 0 else 0,
            avg_corrections_per_execution=sum(m.value for m in corr_metrics) / total if total > 0 else 0,
        )
        
        return stats
    
    def get_metrics_since(self, agent_id: str, minutes: int = 60) -> List[Metric]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return [
            m for m in self.metrics
            if m.agent_id == agent_id and m.timestamp >= cutoff_time
        ]
    
    def get_recent_metrics(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics as dictionaries."""
        agent_metrics = [m for m in self.metrics if m.agent_id == agent_id]
        return [m.to_dict() for m in agent_metrics[-limit:]]
    
    def reset_metrics(self, agent_id: Optional[str] = None) -> None:
        """Reset metrics for an agent or all agents."""
        if agent_id:
            self.metrics = [m for m in self.metrics if m.agent_id != agent_id]
            self.execution_times.pop(agent_id, None)
            self.confidence_scores.pop(agent_id, None)
        else:
            self.metrics.clear()
            self.execution_times.clear()
            self.confidence_scores.clear()
        
        logger.info(f"Reset metrics for agent_id={agent_id or 'all'}")


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
