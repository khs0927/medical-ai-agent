import os
import requests
import json
import logging
from typing import List, Dict, Any, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수에서 API 키 가져오기 (다양한 이름 지원)
HF_API_KEY = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_API_KEY') or os.getenv('HF_API_KEY')
if not HF_API_KEY:
    logger.warning("Hugging Face API 키가 환경 변수에 설정되지 않았습니다. (HUGGINGFACE_TOKEN, HF_TOKEN, HUGGINGFACE_API_KEY, HF_API_KEY)")

class HuggingFaceClient:
    """Hugging Face Inference API와 상호작용하기 위한 클라이언트"""
    def __init__(self, api_key: Optional[str] = HF_API_KEY, base_url: str = "https://api-inference.huggingface.co/models/", timeout: int = 30):
        if not api_key:
            raise ValueError("Hugging Face API 키가 필요합니다.")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def generate_text(
        self,
        model: str,
        inputs: str, # 프롬프트 문자열
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """텍스트 생성을 위해 Hugging Face API를 호출합니다."""
        if parameters is None:
            parameters = {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False # 입력 프롬프트 제외하고 결과만 받기
            }

        payload = {
            "inputs": inputs,
            "parameters": parameters
        }
        api_url = f"{self.base_url}{model}"
        logger.info(f"Hugging Face 텍스트 생성 요청: {model}")

        try:
            response = requests.post(
                api_url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생

            result = response.json()
            logger.info(f"Hugging Face 텍스트 생성 완료: {model}")

            # 결과 형식은 모델마다 다를 수 있음
            if isinstance(result, list) and result:
                generated_text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                 generated_text = result.get('generated_text', '')
            else:
                generated_text = str(result) # 예상치 못한 형식일 경우 문자열 변환

            # Qwen 모델 응답에서 "assistant: " 접두사 제거 (aiModels.ts 참고)
            prefix_to_remove = "assistant: "
            if generated_text.startswith(prefix_to_remove):
                 generated_text = generated_text[len(prefix_to_remove):]

            return generated_text.strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"Hugging Face API 오류 ({model}): {e}")
            if e.response is not None:
                logger.error(f"응답 내용: {e.response.text}")
                # 503 에러 (모델 로딩 중 등) 처리
                if e.response.status_code == 503:
                    try:
                        error_data = e.response.json()
                        estimated_time = error_data.get('estimated_time')
                        if estimated_time:
                             return f"모델 로딩 중입니다. 예상 대기 시간: {estimated_time:.1f}초. 잠시 후 다시 시도해주세요."
                        else:
                             return f"모델을 사용할 수 없습니다 (503 Service Unavailable): {error_data.get('error', '알 수 없는 이유')}"
                    except json.JSONDecodeError:
                         return f"모델을 사용할 수 없습니다 (503 Service Unavailable). 응답 파싱 실패: {e.response.text}"

            return f"오류: Hugging Face API 호출에 실패했습니다 ({e})"
        except Exception as e:
            logger.error(f"텍스트 생성 중 예기치 않은 오류 발생: {e}")
            return f"오류: 응답 처리 중 문제가 발생했습니다."

# 사용 예시 (단독 실행 시)
if __name__ == '__main__':
    import asyncio

    async def main():
        if not HF_API_KEY:
            print("테스트를 위해 환경변수에 HUGGINGFACE_API_KEY를 설정해주세요.")
            return

        client = HuggingFaceClient(api_key=HF_API_KEY)
        model_id = "Qwen/Qwen2.5-Omni-7B" # 사용할 모델 ID (aiModels.ts 참고)

        # 간단한 테스트 프롬프트
        test_prompt = "system: 당신은 도움이 되는 어시스턴트입니다.\n\nuser: 심근경색의 주요 증상은 무엇인가요?"

        print(f"'{model_id}' 모델 테스트 요청 중...")
        response_text = await client.generate_text(model=model_id, inputs=test_prompt)
        print("\n응답:")
        print(response_text)

    asyncio.run(main()) 