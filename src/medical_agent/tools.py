"""
medical_agent.tools
각각 LangGraph 노드로 사용되는 Pure-Python Tool 함수
"""
from __future__ import annotations
from typing import Any, Dict, List
import httpx, os
from pydantic import BaseModel
from .hf_client import qwen_chat, gemini_chat
from .retriever import semantic_retrieval
from .schemas import DrugInteraction, GuidelineSummary, ECGAnalysis, RiskReport

# ────────────────────────────────────────────────────────────
# ECG 분석 (존재 코드 경량화)
# ────────────────────────────────────────────────────────────
def ecg_analysis_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    ecg_payload = state.get("ecg_raw")
    prompt = f"""
    You are a cardiology AI. Analyse this 1-lead ECG JSON and return
    diagnosis, heart_rate, abnormal_flags.
    ECG: {ecg_payload}
    """
    answer = qwen_chat(prompt)
    return {"output": answer}

# ────────────────────────────────────────────────────────────
# 1) 위험 점수
# ────────────────────────────────────────────────────────────
def risk_score_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
    Use TIMI and HEART score criteria to evaluate risk:
    {state['question']}
    """
    return {"output": gemini_chat(prompt)}

# ────────────────────────────────────────────────────────────
# 2) 약물 상호작용 NEW
# ────────────────────────────────────────────────────────────
def drug_interaction_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    drug_list = DrugInteraction.extract_drugs(state["question"])
    resp = httpx.post(
        "https://api.open-medicaments.fr/v1/interactions",
        json={"drugs": drug_list},
        timeout=30,
    ).json()
    return {"output": resp}

# ────────────────────────────────────────────────────────────
# 3) 가이드라인 요약 NEW
# ────────────────────────────────────────────────────────────
def guideline_summary_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    guideline = GuidelineSummary.detect_guideline(state["question"])
    context = semantic_retrieval(guideline)
    answer = gemini_chat(f"Summarise latest guideline:\n{context}")
    return {"output": answer}

# ────────────────────────────────────────────────────────────
# 4) RAG 질문 응답 NEW
# ────────────────────────────────────────────────────────────
def rag_query_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    context = semantic_retrieval(state["question"])
    answer = gemini_chat(f"Context: {context}\n\nQ: {state['question']}\nA:")
    return {"output": answer} 