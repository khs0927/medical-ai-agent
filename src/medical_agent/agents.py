# -*- coding: utf-8 -*-
import logging
from google.adk.agents import Agent

from . import tools
from . import prompts

logger = logging.getLogger(__name__)

# 메인 의료 에이전트 정의
MedicalCoordinatorAgent = Agent(
    name="MedicalCoordinatorAgent",
    # Gemini 모델을 사용하여 사용자 요청 이해 및 도구 호출 조정
    model="gemini-1.5-flash",
    instruction=prompts.HEALTH_CONSULTATION_SYSTEM_PROMPT.format(context=""), # 기본 지침으로 건강 상담 프롬프트 사용
    description="의료 관련 질문에 답하고 ECG 및 건강 데이터 분석 도구를 사용하는 조정 에이전트",
    # 앞서 정의한 도구들을 에이전트가 사용할 수 있도록 추가
    tools=[
        tools.analyze_ecg_data_tool,
        tools.analyze_health_risk_tool
        # 필요시 다른 도구 추가 (예: 웹 검색, 약물 정보 조회 등)
    ],
    # 에이전트가 도구 사용 외 일반적인 답변도 할 수 있도록 설정
    enable_default_response=True,
    # 디버깅을 위한 로그 활성화 (선택 사항)
    verbose=True
)

logger.info(f"{MedicalCoordinatorAgent.name} 에이전트 정의됨.") 