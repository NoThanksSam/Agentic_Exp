"""Hallucination detection and validation system."""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from loguru import logger


class HallucinationRiskLevel(str, Enum):
    """Risk levels for hallucinations."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class HallucinationDetection:
    """Result of hallucination detection."""
    is_hallucinated: bool
    risk_level: HallucinationRiskLevel
    confidence: float  # 0-1, confidence in the detection
    reasons: List[str]
    suspicious_claims: List[str]
    evidence_gaps: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_hallucinated": self.is_hallucinated,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "suspicious_claims": self.suspicious_claims,
            "evidence_gaps": self.evidence_gaps,
        }


class HallucinationDetector:
    """Detects potential hallucinations in agent responses."""
    
    def __init__(self):
        """Initialize hallucination detector."""
        # Patterns that often indicate hallucinations
        self.generic_patterns = [
            r"I don't have.*information",
            r"I couldn't find",
            r"based on my knowledge",
            r"as far as I know",
            r"I believe|I think|I assume",
        ]
        
        # Indicators of confident false statements
        self.confidence_indicators = [
            r"certainly|definitely|absolutely|obviously",
            r"it is well known that",
            r"everyone knows",
            r"there is no doubt",
        ]
        
        logger.info("Initialized HallucinationDetector")
    
    def analyze(
        self,
        response: str,
        context: Optional[str] = None,
        supporting_documents: Optional[List[str]] = None,
    ) -> HallucinationDetection:
        """Analyze response for hallucination indicators."""
        
        reasons: List[str] = []
        suspicious_claims: List[str] = []
        evidence_gaps: List[str] = []
        risk_score = 0.0
        
        # Check 1: High confidence + uncertain knowledge base
        if self._has_high_confidence_language(response):
            risk_score += 0.2
            reasons.append("High confidence language used")
            suspicious_claims.extend(
                self._extract_confident_statements(response)
            )
        
        # Check 2: Specific numbers without sourcing
        specific_claims = self._extract_specific_numbers(response)
        if specific_claims and not context:
            risk_score += 0.15
            reasons.append("Specific numbers without context")
            evidence_gaps.extend(specific_claims)
        
        # Check 3: Claims not in supporting documents
        if supporting_documents:
            unsupported = self._find_unsupported_claims(
                response,
                supporting_documents
            )
            if unsupported:
                risk_score += len(unsupported) * 0.1
                reasons.append("Claims potentially unsupported by documents")
                evidence_gaps.extend(unsupported[:3])  # Limit to 3
        
        # Check 4: Contradiction detection
        contradictions = self._detect_contradictions(response)
        if contradictions:
            risk_score += len(contradictions) * 0.15
            reasons.append("Internal contradictions detected")
            suspicious_claims.extend(contradictions)
        
        # Check 5: Vague uncertainty language
        vague_claims = self._extract_vague_claims(response)
        if len(vague_claims) > 3:
            risk_score += 0.1
            reasons.append("Excessive vague/uncertain language")
        
        # Cap risk score at 1.0 and determine level
        risk_score = min(risk_score, 1.0)
        risk_level = self._score_to_level(risk_score)
        is_hallucinated = risk_score >= 0.5
        
        return HallucinationDetection(
            is_hallucinated=is_hallucinated,
            risk_level=risk_level,
            confidence=min(risk_score, 0.95),
            reasons=reasons,
            suspicious_claims=suspicious_claims[:5],
            evidence_gaps=evidence_gaps[:5],
        )
    
    def _has_high_confidence_language(self, text: str) -> bool:
        """Check if text uses high confidence language."""
        for pattern in self.confidence_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _extract_confident_statements(self, text: str) -> List[str]:
        """Extract statements with high confidence language."""
        sentences = re.split(r'[.!?]+', text)
        claims = [
            s.strip() for s in sentences
            if any(re.search(p, s, re.IGNORECASE) for p in self.confidence_indicators)
            and len(s.strip()) > 10
        ]
        return claims[:3]
    
    def _extract_specific_numbers(self, text: str) -> List[str]:
        """Extract specific numerical claims."""
        numbers = re.findall(r'(\d+\.?\d*\s*(?:%|percent|million|billion|thousand))', text)
        if numbers:
            # Find the sentences containing these numbers
            sentences = re.split(r'[.!?]+', text)
            claims = [
                s.strip() for s in sentences
                if any(num in s for num in numbers)
            ]
            return claims[:3]
        return []
    
    def _find_unsupported_claims(
        self,
        response: str,
        documents: List[str]
    ) -> List[str]:
        """Find claims not supported by documents."""
        sentences = re.split(r'[.!?]+', response)
        unsupported = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Check if any document contains key terms from sentence
            words = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
            if words:
                supported = any(
                    sum(1 for word in words if word in doc.lower()) >= 2
                    for doc in documents
                )
                if not supported:
                    unsupported.append(sentence[:80])
        
        return unsupported
    
    def _detect_contradictions(self, text: str) -> List[str]:
        """Detect internal contradictions."""
        contradictions = []
        
        # Check for common contradiction patterns
        patterns = [
            (r"on one hand|on the other hand", "contradictory statements"),
            (r"while|however|but\s+\w+\s+is\s+opposite", "opposing claims"),
            (r"(\w+)\s+is\s+(\w+)\s+and\s+also\s+(\w+)", "simultaneous opposite states"),
        ]
        
        for pattern, desc in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                contradictions.append(desc)
        
        return contradictions
    
    def _extract_vague_claims(self, text: str) -> List[str]:
        """Extract vague or uncertain claims."""
        vague_patterns = [
            r"might|could|may|possibly|perhaps",
            r"seems|appears|seems to be",
            r"somewhat|kind of|sort of|rather",
        ]
        
        sentences = re.split(r'[.!?]+', text)
        vague = [
            s.strip() for s in sentences
            if any(re.search(p, s, re.IGNORECASE) for p in vague_patterns)
            and len(s.strip()) > 10
        ]
        return vague
    
    def _score_to_level(self, score: float) -> HallucinationRiskLevel:
        """Convert confidence score to risk level."""
        if score < 0.25:
            return HallucinationRiskLevel.NONE
        elif score < 0.5:
            return HallucinationRiskLevel.LOW
        elif score < 0.75:
            return HallucinationRiskLevel.MEDIUM
        else:
            return HallucinationRiskLevel.HIGH


# Global detector instance
_hallucination_detector: Optional[HallucinationDetector] = None


def get_hallucination_detector() -> HallucinationDetector:
    """Get or create hallucination detector."""
    global _hallucination_detector
    if _hallucination_detector is None:
        _hallucination_detector = HallucinationDetector()
    return _hallucination_detector
