import os
import logging
from huggingface_hub import InferenceClient
import google.generativeai as genai

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 설정
QWEN_MODEL = "Qwen/Qwen2.5-Omni-7B"

# HuggingFace 환경 변수
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("HF_TOKEN 환경 변수가 설정되지 않았습니다.")

# Gemini 환경 변수
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    
# HuggingFace 클라이언트 초기화
HF = InferenceClient(model=QWEN_MODEL, token=HF_TOKEN)

# Gemini 클라이언트 초기화
genai.configure(api_key=GEMINI_API_KEY)
GEMINI = genai.GenerativeModel("gemini-pro-1.0")

def qwen_chat(prompt: str) -> str:
    """Qwen 모델을 사용하여 텍스트 생성"""
    try:
        return HF.text_generation(prompt, max_new_tokens=512)
    except Exception as e:
        logger.error(f"Qwen 모델 호출 오류: {e}")
        return f"오류: Qwen 모델 호출에 실패했습니다 ({e})"

def gemini_chat(prompt: str) -> str:
    """Gemini 모델을 사용하여 텍스트 생성"""
    try:
        return GEMINI.generate_content(prompt).text
    except Exception as e:
        logger.error(f"Gemini 모델 호출 오류: {e}")
        return f"오류: Gemini 모델 호출에 실패했습니다 ({e})"

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