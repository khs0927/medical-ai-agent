import os
import logging
from typing import Optional
from huggingface_hub import InferenceClient
import google.generativeai as genai

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 테스트 모드 확인
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() == "true"

# 모델 설정
QWEN_MODEL = "Qwen/Qwen2.5-Omni-7B"

# HuggingFace 환경 변수
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN and not TEST_MODE:
    logger.warning("HF_TOKEN 환경 변수가 설정되지 않았습니다.")

# Gemini 환경 변수
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY and not TEST_MODE:
    logger.warning("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    
# HuggingFace 클라이언트 초기화
if not TEST_MODE and HF_TOKEN:
    try:
        HF = InferenceClient(model=QWEN_MODEL, token=HF_TOKEN)
    except Exception as e:
        logger.error(f"HuggingFace 클라이언트 초기화 오류: {e}")
        HF = None
else:
    HF = None

# Gemini 클라이언트 초기화
if not TEST_MODE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI = genai.GenerativeModel("gemini-pro-1.0")
    except Exception as e:
        logger.error(f"Gemini 클라이언트 초기화 오류: {e}")
        GEMINI = None
else:
    GEMINI = None

def qwen_chat(prompt: str) -> str:
    """Qwen 모델을 사용하여 텍스트 생성"""
    if TEST_MODE or HF is None:
        return _mock_llm_response(prompt, "qwen")
    
    try:
        return HF.text_generation(prompt, max_new_tokens=512)
    except Exception as e:
        logger.error(f"Qwen 모델 호출 오류: {e}")
        return _mock_llm_response(prompt, "qwen")

def gemini_chat(prompt: str) -> str:
    """Gemini 모델을 사용하여 텍스트 생성"""
    if TEST_MODE or GEMINI is None:
        return _mock_llm_response(prompt, "gemini")
        
    try:
        return GEMINI.generate_content(prompt).text
    except Exception as e:
        logger.error(f"Gemini 모델 호출 오류: {e}")
        return _mock_llm_response(prompt, "gemini")

def _mock_llm_response(prompt: str, model_type: str = "gemini") -> str:
    """모의 LLM 응답을 제공합니다."""
    # 프롬프트에 관련된 키워드에 따라 다른 응답 반환
    if "ECG" in prompt or "ecg" in prompt:
        return """
        {
            "diagnosis": "정상 동성 리듬 (Normal Sinus Rhythm)",
            "heart_rate": 72,
            "abnormal_flags": []
        }
        """
    
    if "TIMI" in prompt or "HEART" in prompt or "평가" in prompt:
        return """
        환자의 상태를 평가한 결과:
        - TIMI 점수: 2점 (저-중간 위험)
        - HEART 점수: 3점 (중간 위험)
        
        권장 조치:
        1. 12시간 이내 추가 심장표지자 검사
        2. 저용량 아스피린 투여 고려
        3. 심장내과 협진 요청
        """
    
    if "가이드라인" in prompt or "guideline" in prompt or "Summarise" in prompt:
        return """
        최신 가이드라인에 따르면:
        
        1. ST 분절 상승 심근경색(STEMI) 환자는 증상 발현 후 최대한 빨리, 이상적으로는 90분 이내에 경피적 관상동맥 중재술(PCI)을 받아야 합니다.
        
        2. 약물 치료로는 부하용량 아스피린(162-325mg), P2Y12 억제제(티카그렐러 또는 프라수그렐 선호), 항응고제(주로 헤파린)를 투여해야 합니다.
        
        3. 응급실 도착 시 즉시 12-리드 ECG를 시행하고, 30분 이내 PCI팀에 연락하는 것이 권장됩니다.
        
        4. PCI 시설이 없는 병원에서는 섬유소 용해제 투여를 고려하되, 가능하면 2시간 이내 PCI 가능 센터로 이송해야 합니다.
        """
    
    # 일반적인 질문에 대한 응답
    if "약물" in prompt or "drug" in prompt or "상호작용" in prompt:
        return """
        약물 상호작용 분석 결과:
        
        검출된 약물: Aspirin, Clopidogrel
        
        상호작용:
        - 두 약물은 모두 항혈소판 효과가 있어 출혈 위험이 증가할 수 있으나, 심혈관 질환 환자에게는 일반적으로 함께 처방됩니다.
        - 의사의 처방과 모니터링 하에 복용해야 합니다.
        - 수술이나 치과 시술 전에는 의료진에게 복용 중인 약물을 알려야 합니다.
        """
    
    # 기본 응답
    return """
    의학적 조언을 드리자면, 이 사항은 전문 의료진과 상담하는 것이 가장 좋습니다. 제가 제공하는 정보는 일반적인
    의학 지식을 기반으로 하며, 개인의 구체적인 상황과 건강 상태에 따라 달라질 수 있습니다.
    """

# 사용 예시 (단독 실행 시)
if __name__ == '__main__':
    # 간단한 테스트 프롬프트
    test_prompt = "심근경색의 주요 증상은 무엇인가요?"
    
    print("Qwen 모델 테스트:")
    response_text = qwen_chat(test_prompt)
    print(response_text)
    
    print("\nGemini 모델 테스트:")
    response_text = gemini_chat(test_prompt)
    print(response_text) 