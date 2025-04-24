from pydantic import BaseModel, Field, field_validator
from typing import List
import re

class ConsultationRequest(BaseModel):
    question: str

class ConsultationResponse(BaseModel):
    answer: str

class DrugInteraction(BaseModel):
    drugs: List[str]

    @staticmethod
    def extract_drugs(text: str) -> List[str]:
        return re.findall(r"\b[A-Z][a-z]{2,30}\b", text)

class GuidelineSummary(BaseModel):
    guideline: str

    @staticmethod
    def detect_guideline(text: str) -> str:
        if "STEMI" in text.upper():
            return "ESC 2024 STEMI"
        return "AHA 2023 Chest Pain"

class ECGAnalysis(BaseModel):
    diagnosis: str
    heart_rate: int = Field(gt=0)

class RiskReport(BaseModel):
    timi: int
    heart: int 