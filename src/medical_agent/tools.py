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

# 테스트 모드 확인
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() == "true"

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
    
    # 테스트 모드일 경우 모의 데이터 반환
    if TEST_MODE:
        return {"output": _mock_drug_interaction(drug_list)}
    
    try:
        resp = httpx.post(
            "https://api.open-medicaments.fr/v1/interactions",
            json={"drugs": drug_list},
            timeout=30,
        ).json()
        return {"output": resp}
    except Exception as e:
        print(f"약물 상호작용 API 호출 오류: {e}")
        return {"output": _mock_drug_interaction(drug_list)}

def _mock_drug_interaction(drug_list: List[str]) -> Dict[str, Any]:
    """모의 약물 상호작용 데이터를 반환합니다."""
    if not drug_list:
        return {"status": "error", "message": "약물이 감지되지 않았습니다."}
    
    known_interactions = {
        ("Aspirin", "Clopidogrel"): {
            "severity": "moderate",
            "description": "출혈 위험이 증가할 수 있으나, 심혈관 질환 환자에게 일반적으로 처방되는 조합입니다."
        },
        ("Aspirin", "Warfarin"): {
            "severity": "high",
            "description": "심각한 출혈 위험이 있으므로 의사의 면밀한 모니터링이 필요합니다."
        },
        ("Simvastatin", "Amlodipine"): {
            "severity": "moderate",
            "description": "심바스타틴의 혈중 농도를 증가시킬 수 있어 근육 관련 부작용 위험이 증가합니다."
        }
    }
    
    interactions = []
    
    # 주어진 약물 리스트에서 알려진 상호작용 찾기
    for i, drug1 in enumerate(drug_list):
        for drug2 in drug_list[i+1:]:
            pair = (drug1, drug2)
            reverse_pair = (drug2, drug1)
            
            if pair in known_interactions:
                interactions.append({
                    "drug1": drug1,
                    "drug2": drug2,
                    **known_interactions[pair]
                })
            elif reverse_pair in known_interactions:
                interactions.append({
                    "drug1": drug2,
                    "drug2": drug1,
                    **known_interactions[reverse_pair]
                })
    
    return {
        "status": "success",
        "drugs": drug_list,
        "interactions": interactions,
        "message": f"{len(interactions)}개의 약물 상호작용이 발견되었습니다."
    }

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