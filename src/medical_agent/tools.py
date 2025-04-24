# -*- coding: utf-8 -*-
import json
import logging
from typing import List, Dict, Any, Optional

from google.adk import tool

from .hf_client import HuggingFaceClient
from . import prompts

logger = logging.getLogger(__name__)

# HuggingFace 클라이언트 인스턴스 생성 (API 키는 환경변수에서 로드)
hf_client = HuggingFaceClient()

@tool.tool
def analyze_ecg_data_tool(
    ecg_data: List[float],
    user_info: Dict[str, Any],
    sampling_rate: int = 10
) -> Dict[str, Any]:
    """ECG 데이터를 분석하여 요약, 위험 수준, 권장 사항, 감지된 이슈를 반환합니다.

    Args:
        ecg_data: ECG 데이터 포인트 리스트.
        user_info: 사용자 정보 딕셔너리 (age, gender, medicalConditions (list or None)).
        sampling_rate: ECG 데이터 샘플링 비율 (기본값: 10, 10개 중 1개).

    Returns:
        분석 결과 딕셔너리 또는 오류 메시지 딕셔너리.
    """
    logger.info(f"ECG 데이터 분석 도구 호출됨. 데이터 포인트: {len(ecg_data)}")
    try:
        # ECG 데이터 샘플링 (aiModels.ts 로직 참고)
        sampled_data = ecg_data[::sampling_rate]
        if not sampled_data: # 샘플링 결과 데이터가 없는 경우 원본 사용 고려 또는 오류 반환
            sampled_data = ecg_data[:100] # 예시: 최대 100개 사용
            logger.warning("샘플링된 ECG 데이터가 없습니다. 원본 데이터 일부를 사용합니다.")

        ecg_data_json = json.dumps(sampled_data)

        # 사용자 정보 추출 (기본값 처리)
        age = user_info.get('age', 'N/A')
        gender = user_info.get('gender', 'N/A')
        conditions = user_info.get('medicalConditions')
        medical_conditions = ", ".join(conditions) if conditions else '없음'

        # 프롬프트 생성
        user_prompt = prompts.ECG_ANALYSIS_USER_PROMPT_TEMPLATE.format(
            age=age,
            gender=gender,
            medical_conditions=medical_conditions,
            ecg_data_json=ecg_data_json
        )
        full_prompt = prompts.format_qwen_prompt(prompts.ECG_ANALYSIS_SYSTEM_PROMPT, user_prompt)

        # Hugging Face API 호출 (비동기 방식 사용 필요 시 hf_client 수정 필요)
        # 현재 ADK tool 데코레이터는 동기 함수를 가정하는 것으로 보임
        # TODO: ADK tool에서 비동기 I/O 지원 여부 확인 및 필요 시 hf_client 수정
        # 여기서는 일단 동기적으로 호출하는 것처럼 작성 (실제로는 비동기 필요)
        response_text = hf_client.generate_text(model=prompts.QWEN_MEDICAL_MODEL, inputs=full_prompt)

        # JSON 응답 파싱
        try:
            # 응답 텍스트에서 JSON 부분만 추출 (모델이 추가 텍스트를 반환할 수 있음)
            json_match = json.loads(response_text)
            if isinstance(json_match, dict):
                 logger.info("ECG 분석 결과 수신 및 파싱 성공")
                 return json_match
            else:
                raise ValueError("파싱된 결과가 딕셔너리 형태가 아님")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"ECG 분석 결과 파싱 실패: {e}. 원본 응답: {response_text}")
            # 모델이 JSON 형식으로 응답하지 못했을 경우, 텍스트 자체를 반환 시도
            return {"error": "결과 파싱 실패", "raw_response": response_text}

    except Exception as e:
        logger.exception(f"ECG 분석 도구 실행 중 오류 발생: {e}")
        return {"error": f"ECG 분석 중 오류 발생: {e}"}

@tool.tool
def analyze_health_risk_tool(health_data: Dict[str, Any]) -> Dict[str, Any]:
    """종합적인 건강 데이터를 분석하여 위험 점수, 위험 요인, 제안 사항을 반환합니다.

    Args:
        health_data: 건강 데이터 딕셔너리 (heartRate, bloodPressureSystolic,
            bloodPressureDiastolic, oxygenLevel, temperature, age, gender,
            riskFactors (list or None)).

    Returns:
        위험 분석 결과 딕셔너리 또는 오류 메시지 딕셔너리.
    """
    logger.info("건강 위험 분석 도구 호출됨")
    try:
        # 건강 데이터 추출 (기본값 처리)
        heart_rate = health_data.get('heartRate', 'N/A')
        systolic_bp = health_data.get('bloodPressureSystolic', 'N/A')
        diastolic_bp = health_data.get('bloodPressureDiastolic', 'N/A')
        oxygen_level = health_data.get('oxygenLevel', 'N/A')
        temperature = health_data.get('temperature', 'N/A')
        age = health_data.get('age', 'N/A')
        gender = health_data.get('gender', 'N/A')
        factors = health_data.get('riskFactors')
        risk_factors = ", ".join(factors) if factors else '없음'

        # 프롬프트 생성
        user_prompt = prompts.HEALTH_RISK_ANALYSIS_USER_PROMPT_TEMPLATE.format(
            heart_rate=heart_rate,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            oxygen_level=oxygen_level,
            temperature=temperature,
            age=age,
            gender=gender,
            risk_factors=risk_factors
        )
        full_prompt = prompts.format_qwen_prompt(prompts.HEALTH_RISK_ANALYSIS_SYSTEM_PROMPT, user_prompt)

        # Hugging Face API 호출 (동기 가정)
        response_text = hf_client.generate_text(model=prompts.QWEN_MEDICAL_MODEL, inputs=full_prompt)

        # JSON 응답 파싱
        try:
            json_match = json.loads(response_text)
            if isinstance(json_match, dict):
                logger.info("건강 위험 분석 결과 수신 및 파싱 성공")
                return json_match
            else:
                raise ValueError("파싱된 결과가 딕셔너리 형태가 아님")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"건강 위험 분석 결과 파싱 실패: {e}. 원본 응답: {response_text}")
            return {"error": "결과 파싱 실패", "raw_response": response_text}

    except Exception as e:
        logger.exception(f"건강 위험 분석 도구 실행 중 오류 발생: {e}")
        return {"error": f"건강 위험 분석 중 오류 발생: {e}"} 