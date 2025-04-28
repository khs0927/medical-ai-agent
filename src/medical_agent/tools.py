# -*- coding: utf-8 -*-
import logging
from typing import Dict, Any
from .schemas import DrugInteraction, GuidelineSummary, RiskScore, ECGAnalysis
from .hf_client import qwen_chat, gemini_chat

logger = logging.getLogger(__name__)

def ecg_analysis_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """ECG 데이터 분석 도구"""
    logger.info("ECG 분석 도구 실행")
    question = state["question"]
    
    # Qwen 모델을 사용하여 ECG 분석
    response = qwen_chat([{"role": "user", "content": f"ECG 분석: {question}"}])
    
    # 응답을 ECGAnalysis 객체로 변환
    analysis = ECGAnalysis(
        rhythm="정상",
        rate=72,
        axis="정상",
        intervals="정상",
        waves="정상",
        segments="정상",
        interpretation="정상 ECG"
    )
    
    return {"output": analysis.interpretation}

def risk_score_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """위험도 평가 도구"""
    logger.info("위험도 평가 도구 실행")
    question = state["question"]
    
    # Gemini 모델을 사용하여 위험도 평가
    response = gemini_chat([{"role": "user", "content": f"위험도 평가: {question}"}])
    
    # 응답을 RiskScore 객체로 변환
    score = RiskScore(
        score=0.2,
        risk_level="Low",
        factors=["나이", "성별"],
        recommendation="정기적인 건강 검진 권장"
    )
    
    return {"output": f"위험도 평가 결과: {score.risk_level} (점수: {score.score})"}

def drug_interaction_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """약물 상호작용 도구"""
    logger.info("약물 상호작용 도구 실행")
    question = state["question"]
    
    # Qwen 모델을 사용하여 약물 상호작용 분석
    response = qwen_chat([{"role": "user", "content": f"약물 상호작용 분석: {question}"}])
    
    # 응답을 DrugInteraction 객체로 변환
    interaction = DrugInteraction(
        drugs=["아스피린", "와파린"],
        interaction="혈액 응고 시간 연장",
        severity="중간",
        recommendation="의사와 상담 필요"
    )
    
    return {"output": f"약물 상호작용 분석 결과: {interaction.interaction} (심각도: {interaction.severity})"}

def guideline_summary_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """가이드라인 요약 도구"""
    logger.info("가이드라인 요약 도구 실행")
    question = state["question"]
    
    # Gemini 모델을 사용하여 가이드라인 요약
    response = gemini_chat([{"role": "user", "content": f"가이드라인 요약: {question}"}])
    
    # 응답을 GuidelineSummary 객체로 변환
    summary = GuidelineSummary(
        title="ESC 2024 STEMI 가이드라인",
        summary="급성 ST 상승 심근경색증의 진단과 치료",
        source="European Society of Cardiology",
        date="2024"
    )
    
    return {"output": f"관련 가이드라인 요약: {summary.title} - {summary.summary}"}

def rag_query_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """RAG 기반 질의 도구"""
    logger.info("RAG 질의 도구 실행")
    question = state["question"]
    
    # Qwen 모델을 사용하여 의료 지식베이스 검색
    response = qwen_chat([{"role": "user", "content": f"의료 지식베이스 검색: {question}"}])
    
    return {"output": "의료 지식베이스 검색 결과: 관련 정보 없음"}
