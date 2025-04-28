# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import re

class ConsultationRequest(BaseModel):
    """상담 요청"""
    question: str

class ConsultationResponse(BaseModel):
    """상담 응답"""
    answer: str

class ConsultationState(BaseModel):
    """상담 상태"""
    question: str
    tool: Optional[str] = None
    output: Optional[str] = None

class DrugInteraction(BaseModel):
    """약물 상호작용"""
    drugs: List[str]
    interaction: str
    severity: str
    recommendation: str

class GuidelineSummary(BaseModel):
    """가이드라인 요약"""
    title: str
    summary: str
    source: str
    date: str

class ECGAnalysis(BaseModel):
    diagnosis: str
    heart_rate: int = Field(gt=0)

class RiskReport(BaseModel):
    timi: int
    heart: int

class RiskScore(BaseModel):
    """위험도 평가"""
    score: float
    risk_level: str
    factors: List[str]
    recommendation: str

    @staticmethod
    def detect_risk_level(score: float) -> str:
        if score < 0.2:
            return "Low"
        elif score < 0.5:
            return "Moderate"
        else:
            return "High"

    @staticmethod
    def extract_factors(text: str) -> List[str]:
        return re.findall(r"\b[A-Z][a-z]{2,30}\b", text) 