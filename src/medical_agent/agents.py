# -*- coding: utf-8 -*-
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END, START
from src.medical_agent.tools import (
    ecg_analysis_tool,
    risk_score_tool,
    drug_interaction_tool,
    guideline_summary_tool,
    rag_query_tool,
)
from src.medical_agent.schemas import ConsultationRequest, ConsultationResponse, ConsultationState
from src.medical_agent.hf_client import qwen_chat, gemini_chat

logger = logging.getLogger(__name__)

def _planner(state: Dict[str, Any]) -> Dict[str, Any]:
    """1단계: 요청 의도 파악 → 사용할 tool 리스트 반환"""
    question = state["question"]
    
    if "약물" in question or "drug" in question:
        state["output"] = drug_interaction_tool(state)["output"]
    elif "가이드라인" in question:
        state["output"] = guideline_summary_tool(state)["output"]
    elif any(k in question for k in ("위험", "risk", "심박")):
        state["output"] = risk_score_tool(state)["output"]
    else:
        state["output"] = rag_query_tool(state)["output"]
    
    return state

graph = StateGraph(state_schema=ConsultationState)
graph.add_node("plan", _planner)

# 엣지 추가
graph.add_edge(START, "plan")
graph.add_edge("plan", END)

consultation_agent = graph.compile()

def consult(req: ConsultationRequest) -> ConsultationResponse:
    result = consultation_agent.invoke({"question": req.question})
    return ConsultationResponse(answer=result["output"])

logger.info("LangGraph 기반 의료 에이전트 정의됨.") 