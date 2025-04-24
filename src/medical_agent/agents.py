# -*- coding: utf-8 -*-
import logging
from langgraph.graph import StateGraph, END
from .tools import (
    ecg_analysis_tool,
    risk_score_tool,
    drug_interaction_tool,
    guideline_summary_tool,
    rag_query_tool,
)
from .schemas import ConsultationRequest, ConsultationResponse
from .hf_client import qwen_chat, gemini_chat

logger = logging.getLogger(__name__)

def _planner(state):
    """1단계: 요청 의도 파악 → 사용할 tool 리스트 반환"""
    question = state["question"]
    if "약물" in question or "drug" in question:
        return {"tool": "drug_interaction_tool"}
    if "가이드라인" in question:
        return {"tool": "guideline_summary_tool"}
    if any(k in question for k in ("위험", "risk", "심박")):
        return {"tool": "risk_score_tool"}
    return {"tool": "rag_query_tool"}

graph = StateGraph()
graph.add_node("plan", _planner)
graph.add_node("drug_interaction_tool", drug_interaction_tool)
graph.add_node("guideline_summary_tool", guideline_summary_tool)
graph.add_node("risk_score_tool", risk_score_tool)
graph.add_node("rag_query_tool", rag_query_tool)

graph.add_edge("plan", "drug_interaction_tool", condition=lambda s: s["tool"]=="drug_interaction_tool")
graph.add_edge("plan", "guideline_summary_tool", condition=lambda s: s["tool"]=="guideline_summary_tool")
graph.add_edge("plan", "risk_score_tool", condition=lambda s: s["tool"]=="risk_score_tool")
graph.add_edge("plan", "rag_query_tool", condition=lambda s: s["tool"]=="rag_query_tool")
graph.add_edge("drug_interaction_tool", END)
graph.add_edge("guideline_summary_tool", END)
graph.add_edge("risk_score_tool", END)
graph.add_edge("rag_query_tool", END)

consultation_agent = graph.compile()

def consult(req: ConsultationRequest) -> ConsultationResponse:
    result = consultation_agent.invoke({"question": req.question})
    return ConsultationResponse(answer=result["output"])

logger.info("LangGraph 기반 의료 에이전트 정의됨.") 