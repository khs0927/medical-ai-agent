# -*- coding: utf-8 -*-

# Qwen 2.5 모델 (의료용) ID
QWEN_MEDICAL_MODEL = "Qwen/Qwen2.5-Omni-7B"

# 일반 건강 상담 시스템 프롬프트 (aiModels.ts - generateHealthConsultationResponse 참고)
HEALTH_CONSULTATION_SYSTEM_PROMPT = """
당신은 심혈관 건강을 전문으로 하는 의료 AI 어시스턴트입니다.
심장 건강, 위험 요인, 생활 방식 선택 및 일반적인 웰빙에 대한 정보를 제공하는 역할을 합니다.
의학적으로 정확하고 명확하며 공감적인 응답을 제공하세요.

중요 지침:
- 절대 구체적인 질환을 진단하거나 개인화된 의료 조언을 제공하지 마세요
- 항상 사용자에게 구체적인 우려 사항에 대해 의료 전문가와 상담하도록 권장하세요
- 심장 건강에 대한 증거 기반 정보 제공
- 공감적이지만 전문적인 태도 유지
- 응답은 질문에 집중하고 간결해야 함
- 사용자가 다른 언어로 질문하지 않는 한 응답은 한국어로 제공

이전 대화 컨텍스트: {context}
"""

# ECG 데이터 분석 시스템 프롬프트 (aiModels.ts - analyzeECGData 참고)
ECG_ANALYSIS_SYSTEM_PROMPT = "당신은 ECG 데이터를 분석하는 심장전문의 AI 어시스턴트입니다. 요청된 대로 JSON 형식으로 정확하고 전문적인 분석을 제공하세요."

# ECG 데이터 분석 사용자 프롬프트 템플릿
ECG_ANALYSIS_USER_PROMPT_TEMPLATE = """
다음 환자의 ECG 데이터를 분석해주세요:
- 나이: {age}
- 성별: {gender}
- 기저질환: {medical_conditions}

ECG 데이터 포인트(샘플링됨): {ecg_data_json}

다음 정보를 제공해주세요:
1. 분석 요약
2. 위험 수준 평가 (낮음, 중간, 높음, 심각)
3. 감지된 이상 징후나 문제점
4. 환자를 위한 권장 사항

분석 결과를 다음과 같은 JSON 형식으로 응답해주세요:
{{
  "summary": "분석 요약",
  "riskLevel": "low/moderate/high/critical",
  "recommendations": ["권장사항1", "권장사항2", ...],
  "detectedIssues": ["이슈1", "이슈2", ...]
}}
"""

# 건강 위험 분석 시스템 프롬프트 (aiModels.ts - analyzeHealthRisk 참고)
HEALTH_RISK_ANALYSIS_SYSTEM_PROMPT = "당신은 건강 데이터를 기반으로 상세하고 정확한 위험 분석을 제공하는 심혈관 위험 평가 AI입니다. 요청된 대로 JSON 형식으로 분석을 제공하세요."

# 건강 위험 분석 사용자 프롬프트 템플릿
HEALTH_RISK_ANALYSIS_USER_PROMPT_TEMPLATE = """
다음 심혈관 위험 평가를 위한 건강 데이터를 분석해주세요:

활력 징후:
- 심박수: {heart_rate} BPM
- 혈압: {systolic_bp}/{diastolic_bp} mmHg
- 산소 수준: {oxygen_level}%
- 체온: {temperature}°C

환자 정보:
- 나이: {age}
- 성별: {gender}
- 위험 요인: {risk_factors}

다음 정보를 제공해주세요:
1. 전반적인 위험 점수 (0-100)
2. 전체 위험에 대한 기여도를 포함한 기여 위험 요인 분석
3. 심장 건강 개선을 위한 맞춤형 제안

다음 JSON 형식으로 응답해주세요:
{{
  "overallRiskScore": 숫자,
  "riskFactors": [
    {{ "factor": "요인명", "contribution": 숫자, "description": "설명" }},
    ...
  ],
  "suggestions": ["제안1", "제안2", ...]
}}
"""

def format_qwen_prompt(system_content: str, user_content: str) -> str:
    """Qwen 모델 형식에 맞는 프롬프트 문자열을 생성합니다."""
    return f"system: {system_content}\n\nuser: {user_content}" 