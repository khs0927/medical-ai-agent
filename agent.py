import inspect
import json
import logging
import os
import re
import time
import asyncio
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# --- 초기 설정 ---
# .env 파일 로드 (프로젝트 루트에 있다고 가정)
load_dotenv()

# 로깅 레벨 설정 (환경 변수 또는 기본값 INFO 사용)
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 테스트 모드 설정 (기본값: False)
TEST_MODE = False

# LLM 및 API 클라이언트 라이브러리 로드 시도
try:
    import google.generativeai as genai
    google_ai_available = True
except ImportError:
    logger.warning('Gemini 라이브러리(google-generativeai)가 설치되지 않았습니다. LLM 기능이 제한됩니다.')
    google_ai_available = False

# MedLLaMA 라이브러리 로드 시도
try:
    import torch
    medllama_available = True
except ImportError:
    logger.warning('MedLLaMA 관련 라이브러리(torch, transformers)가 설치되지 않았습니다. Gemini만 사용하여 계속 진행합니다.')
    medllama_available = False

# --- 환경 변수 및 설정 로드 ---
# AI Security Analyzer Backend
AI_ANALYZER_API_ENDPOINT = os.getenv('AI_ANALYZER_API_ENDPOINT', 'http://localhost:8000') # 기본 URL
AI_ANALYZER_API_KEY = os.getenv('AI_ANALYZER_API_KEY')

# LLM Configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini')
LLM_API_KEY = os.getenv('LLM_API_KEY')
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-pro-preview-03-25')

# MedLLaMA 설정
MEDLLAMA_MODEL_NAME = os.getenv('MEDLLAMA_MODEL_NAME', 'JSL-MedLlama-3-8B-v2.0')
MEDLLAMA_MODEL_PATH = os.getenv('MEDLLAMA_MODEL_PATH', '')
MEDLLAMA_DEVICE = os.getenv('MEDLLAMA_DEVICE', 'cuda')

# API Call Settings
DEFAULT_API_TIMEOUT = int(os.getenv('DEFAULT_API_TIMEOUT', '20')) # 초 단위
DEFAULT_API_RETRIES = int(os.getenv('DEFAULT_API_RETRIES', '2')) # 재시도 횟수
DEFAULT_API_RETRY_DELAY = int(os.getenv('DEFAULT_API_RETRY_DELAY', '2')) # 재시도 간격 (초)

# PubMed Settings (백엔드 호출 시 필요)
PUBMED_API_KEY = os.getenv('PUBMED_API_KEY') # 백엔드 호출 시 사용될 수 있음

# --- Mock 데이터 (예시, 실제로는 잘 사용되지 않음) ---
MOCK_ANALYZER_RESPONSE = {'status': 'success_mock', 'analysis_id': 'mock-12345', 'results': {'risk_score': 0.75}}
MOCK_USER_HEALTH_DATA = {'user_id': 'user_mock_001', 'vitals': {'heart_rate': 72}}

# 웹 검색 및 스크래핑 유틸리티 추가
try:
    from utils.web_search import MedicalWebSearch
    from utils.web_scraper import MedicalWebScraper
    from retrievers.web_retriever import WebRetriever
    web_search_available = True
except ImportError:
    web_search_available = False
    logger.warning('웹 검색 모듈을 로드할 수 없습니다. 웹 검색 기능이 비활성화됩니다.')


class MedicalAgentError(Exception):
    '''Medical Agent 관련 오류 처리를 위한 사용자 정의 예외'''


class MedicalAgent:
    '''
    의료 정보를 수집, 분석하고 전문가 수준의 답변을 생성하는 AI 에이전트.
    LLM과 백엔드 API를 활용하여 다양한 도구를 오케스트레이션합니다.
    '''

    def __init__(self, config_path=None):
        """
        MedicalAgent 초기화

        Args:
            config_path (str, optional): 설정 파일 경로. 기본값은 None으로, 기본 설정 파일을 사용합니다.
        """
        # 기본 설정 파일 경로
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config/agent_config.json')
        
        # 설정 로드
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"설정 파일을 로드했습니다: {config_path}")
        except Exception as e:
            logger.error(f"설정 파일을 로드하는 중 오류가 발생했습니다: {e}")
            # 기본 설정 제공
            self.config = {
                "llm": {
                    "type": "gemini",
                    "api_key": None,
                    "model": "gemini-pro"
                },
                "web_search": {
                    "enabled": True,
                    "max_results": 5
                }
            }
            logger.warning("기본 설정을 사용합니다.")
        
        # LLM 설정
        self.llm_config = self.config.get('llm', {})
        
        # 웹 검색 설정
        web_search_config = self.config.get('web_search', {})
        self.web_search_enabled = web_search_config.get('enabled', False)
        self.web_search_max_results = web_search_config.get('max_results', 5)
        self.last_web_sources = []  # 마지막 웹 검색 소스 저장
        
        # 데이터 소스 및 API 초기화
        self.pubmed_client = None
        self.kaggle_client = None
        self.initialize_data_sources()
        
        # 웹 검색 초기화
        self.web_retriever = None
        if self.web_search_enabled:
            try:
                from retrievers.web_retriever import WebRetriever
                self.web_retriever = WebRetriever()
                logger.info("웹 검색 기능이 활성화되었습니다.")
            except ImportError as e:
                logger.warning(f"웹 검색 모듈을 가져오는 중 오류 발생: {e}")
                self.web_search_enabled = False
            except Exception as e:
                logger.warning(f"웹 검색 초기화 중 오류 발생: {e}")
                self.web_search_enabled = False
        
        # 하이브리드 모델 초기화
        self.gemini_model = None
        self.medllama_model = None
        self.initialize_models()
        
        logger.info("의료 에이전트가 초기화되었습니다.")
    
    def initialize_models(self):
        """모델 초기화 - Gemini와 MedLlama 모델 로드"""
        try:
            # Gemini 모델 초기화
            if self.llm_config.get("type") == "gemini" and self.llm_config.get("api_key"):
                import google.generativeai as genai
                genai.configure(api_key=self.llm_config.get("api_key"))
                
                # 최신 Gemini 2.5 Pro 모델 사용 (혹은 사용 가능한 최신 모델)
                model_name = "gemini-1.5-pro" if "gemini-1.5-pro" in genai.list_models() else "gemini-pro"
                self.gemini_model = genai.GenerativeModel(model_name)
                logger.info("Gemini 모델이 초기화되었습니다: " + model_name)
        except Exception as e:
            logger.error("Gemini 모델 초기화 중 오류 발생: " + str(e))
            
        # MedLlama 모델 초기화 (사용 가능한 경우)
        try:
            # 가능하다면 헬스케어 특화 모델 로드
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.medllama_tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b")
            self.medllama_model = AutoModelForCausalLM.from_pretrained("medalpaca/medalpaca-7b")
            logger.info("MedLlama 모델이 초기화되었습니다.")
        except Exception as e:
            logger.warning("MedLlama 모델 초기화 오류 (하이브리드 모드가 제한될 수 있음): " + str(e))
    
    def initialize_data_sources(self):
        """데이터 소스 초기화 - PubMed 및 Kaggle API"""
        # PubMed API 초기화
        try:
            from data_sources.pubmed_api import PubMedAPI
            self.pubmed_client = PubMedAPI()
            logger.info("PubMed API가 초기화되었습니다.")
        except ImportError as e:
            logger.warning("PubMed API 모듈을 가져오는 중 오류 발생: " + str(e))
        except Exception as e:
            logger.warning("PubMed API 초기화 중 오류 발생: " + str(e))
        
        # Kaggle API 초기화
        try:
            from data_sources.kaggle_api import KaggleAPI
            self.kaggle_client = KaggleAPI()
            logger.info("Kaggle API가 초기화되었습니다.")
        except ImportError as e:
            logger.warning("Kaggle API 모듈을 가져오는 중 오류 발생: " + str(e))
        except Exception as e:
            logger.warning("Kaggle API 초기화 중 오류 발생: " + str(e))

    async def direct_web_search(self, query: str, search_type: str = 'medical', num_results: int = 5) -> list:
        """
        웹 검색을 직접 수행하여 관련 정보 검색 (asyncio 사용)

        Args:
            query: 검색 쿼리
            search_type: 검색 유형 ('medical', 'general', 'academic')
            num_results: 반환할 최대 결과 수

        Returns:
            검색 결과 목록 (제목, URL, 스니펫 포함)
        """
        try:
            # 의료 관련 키워드 추가 (의학 검색 시)
            search_query = query
            if search_type == 'medical':
                if not any(kw in query.lower() for kw in ['의학', '의료', '건강', 'medical', 'health', 'clinical']):
                    search_query = f"의학 정보 {query}"
            
            # 웹 검색 수행
            results = []
            if web_search_available:
                try:
                    web_search = MedicalWebSearch()
                    results = await web_search.search(search_query, max_results=num_results)
                except Exception as e:
                    logger.error(f"웹 검색 중 오류 발생: {e}", exc_info=True)
            
            return results
        except Exception as e:
            logger.error(f'직접 웹 검색 중 오류 발생: {e}', exc_info=True)
            return []

    def process_query(self, query: str, model_name: str = 'hybrid-medical', conversation_history: list = None, use_pubmed: bool = True, use_kaggle: bool = False) -> str:
        """사용자 쿼리 처리 - 의료 정보 생성"""
        try:
            if not query:
                return "질문을 입력해주세요."
            
            logger.info("쿼리 처리 시작: " + query[:50] + "...")
            
            # 대화 기록이 없으면 초기화
            if conversation_history is None:
                conversation_history = []
            
            # 모델 선택
            if model_name == 'hybrid-medical':
                if self.gemini_model and self.medllama_model:
                    logger.info("하이브리드 의료 모델 사용 (Gemini + MedLlama)")
                elif self.gemini_model:
                    logger.info("Gemini 모델만 사용")
                    model_name = 'gemini'
                elif self.medllama_model:
                    logger.info("MedLlama 모델만 사용")
                    model_name = 'med-llama'
                else:
                    logger.warning("사용 가능한 모델이 없습니다. 기본 응답을 반환합니다.")
                    return "죄송합니다. 현재 AI 모델을 사용할 수 없습니다. 나중에 다시 시도해주세요."
            
            # 시작 시간 기록
            start_time = time.time()
            
            # 1. 사용자 질문 분석
            logger.info("사용자 질문 분석 중...")
            
            # 2. 관련 의학 정보 검색
            logger.info("관련 의학 정보 검색 중...")
            medical_context = []
            
            # 3. PubMed 검색 (의학 문헌)
            if use_pubmed and self.pubmed_client and not TEST_MODE:
                try:
                    logger.info("PubMed 검색 수행 중...")
                    # PubMed API 검색
                    pubmed_results = self.pubmed_client.search(query, max_results=5)
                    if pubmed_results:
                        pubmed_context = "\n\n".join([
                            f"제목: {result.get('title', 'N/A')}\n"
                            f"초록: {result.get('abstract', 'N/A')}\n"
                            f"저널: {result.get('journal', 'N/A')}\n"
                            f"URL: {result.get('url', 'N/A')}"
                            for result in pubmed_results
                        ])
                        medical_context.append(f"### PubMed 의학 문헌 검색 결과:\n{pubmed_context}")
                        logger.info(f"PubMed 검색 완료: {len(pubmed_results)}개 결과 발견")
                except Exception as e:
                    logger.error(f"PubMed 검색 중 오류 발생: {e}", exc_info=True)
            
            # 4. 웹 검색 결과 (의학 정보)
            if not TEST_MODE:
                try:
                    logger.info("의학 정보 웹 검색 수행 중...")
                    web_results = asyncio.run(self.direct_web_search(query, search_type='medical'))
                    self.last_web_sources = web_results
                    
                    if web_results:
                        web_context = "\n\n".join([
                            f"제목: {result.get('title', 'N/A')}\n"
                            f"내용: {result.get('snippet', 'N/A')}\n"
                            f"URL: {result.get('url', 'N/A')}"
                            for result in web_results
                        ])
                        medical_context.append(f"### 의학 웹 검색 결과:\n{web_context}")
                        logger.info(f"웹 검색 완료: {len(web_results)}개 결과 발견")
                except Exception as e:
                    logger.error(f"웹 검색 중 오류 발생: {e}", exc_info=True)
            
            # Kaggle 데이터셋 검색 (선택적)
            if use_kaggle and self.kaggle_client and not TEST_MODE:
                try:
                    logger.info("Kaggle 데이터셋 검색 수행 중...")
                    # Kaggle API 검색
                    kaggle_results = self.kaggle_client.search_datasets(query, max_results=3)
                    if kaggle_results:
                        kaggle_context = "\n\n".join([
                            f"제목: {result.get('title', 'N/A')}\n"
                            f"설명: {result.get('description', 'N/A')}\n"
                            f"URL: {result.get('url', 'N/A')}"
                            for result in kaggle_results
                        ])
                        medical_context.append(f"### Kaggle 의학 데이터셋 검색 결과:\n{kaggle_context}")
                        logger.info(f"Kaggle 검색 완료: {len(kaggle_results)}개 결과 발견")
                except Exception as e:
                    logger.error(f"Kaggle 검색 중 오류 발생: {e}", exc_info=True)
            
            # 대화 컨텍스트 추가
            if conversation_history:
                conversation_context = "\n".join([
                    f"사용자: {msg['user']}\nAI: {msg['ai']}"
                    for msg in conversation_history[-3:]  # 최근 3개 대화만 사용
                ])
                medical_context.append(f"### 이전 대화 컨텍스트:\n{conversation_context}")
            
            # 의학 문헌 검색 결과 정리
            if medical_context:
                medical_info = "\n\n".join(medical_context)
            else:
                medical_info = ""
            
            # 5. 프롬프트 구성
            prompt = """
            당신은 '메디컬 AI'라는 전문 의료 보조 에이전트��니다.
            
            다음 질문에 대해 정확하고 신뢰할 수 있는 의학 정보를 제공해주세요. 
            의학적으로 검증된 정보만 제공하고, 확실하지 않은 내용은 명시해주세요.
            
            구조화된 응답 형식으로 답변해주세요:
            
            1. 질문 의도 파악
            2. 관련 의학 정보
            3. 권장 조치
            4. 신뢰도 평가 (상, 중, 하)
            
            질문: """ + query + """
            
            """ + medical_info
            
            # 6. LLM 추론 실행
            response = None
            
            # 선택된 모델 사용
            if model_name == 'hybrid-medical':
                # 하이브리드 모드 - Gemini와 MedLlama 모델 모두 사용
                if self.gemini_model and self.medllama_model:
                    try:
                        # Gemini 모델 먼저 사용
                        gemini_response = self.generate_with_gemini(prompt)
                        logger.info("Gemini 모델 응답 생성 완료")
                        
                        # MedLlama 모델로 추가 검증
                        med_validation_prompt = """
                        다음은 의학 질문과 Gemini AI 모델이 생성한 응답입니다.
                        이 응답이 의학적으로 정확한지 평가하고, 필요시 수정해주세요.
                        
                        질문: """ + query + """
                        
                        Gemini 응답:
                        """ + gemini_response + """
                        
                        의학적 정확성 검증 및 수정 (잘못된 정보는 수정하되, 구조는 유지):
                        """
                        
                        med_llama_response = self.generate_with_medllama(med_validation_prompt)
                        logger.info("MedLlama 모델 검증 응답 생성 완료")
                        
                        # 최종 응답 구성 (하이브리드)
                        response = med_llama_response
                    except Exception as e:
                        logger.error(f"하이브리드 모델 응답 생성 중 오류: {e}", exc_info=True)
                        response = "죄송합니다. AI 모델 처리 중 오류가 발생했습니다. 다시 시도해주세요."
                
                # Gemini 모델만 사용 가능한 경우
                elif self.gemini_model:
                    try:
                        response = self.generate_with_gemini(prompt)
                        logger.info("Gemini 모델 응답 생성 완료")
                    except Exception as e:
                        logger.error(f"Gemini 모델 응답 생성 중 오류: {e}", exc_info=True)
                        response = "죄송합니다. Gemini 모델 처리 중 오류가 발생했습니다. 다시 시도해주세요."
                
                # MedLlama 모델만 사용 가능한 경우
                elif self.medllama_model:
                    try:
                        response = self.generate_with_medllama(prompt)
                        logger.info("MedLlama 모델 응답 생성 완료")
                    except Exception as e:
                        logger.error(f"MedLlama 모델 응답 생성 중 오류: {e}", exc_info=True)
                        response = "죄송합니다. MedLlama 모델 처리 중 오류가 발생했습니다. 다시 시도해주세요."
            
            # Gemini 모델 전용 모드
            elif model_name == 'gemini' and self.gemini_model:
                try:
                    response = self.generate_with_gemini(prompt)
                    logger.info("Gemini 모델 응답 생성 완료")
                except Exception as e:
                    logger.error(f"Gemini 모델 응답 생성 중 오류: {e}", exc_info=True)
                    response = "죄송합니다. Gemini 모델 처리 중 오류가 발생했습니다. 다시 시도해주세요."
            
            # MedLlama 모델 전용 모드
            elif model_name == 'med-llama' and self.medllama_model:
                try:
                    response = self.generate_with_medllama(prompt)
                    logger.info("MedLlama 모델 응답 생성 완료")
                except Exception as e:
                    logger.error(f"MedLlama 모델 응답 생성 중 오류: {e}", exc_info=True)
                    response = "죄송합니다. MedLlama 모델 처리 중 오류가 발생했습니다. 다시 시도해주세요."
            
            else:
                logger.error("지원되지 않는 모델: " + model_name)
                response = "죄송합니다. 요청하신 AI 모델이 현재 지원되지 않거나 사용할 수 없습니다."
            
            # 7. 소요 시간 계산 및 로깅
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"쿼리 처리 완료, 소요 시간: {elapsed_time:.2f}초")
            
            # 8. 웹 출처 정보 추가 (실제 데이터 소스를 사용한 경우만)
            if self.last_web_sources and not TEST_MODE:
                source_section = "\n\n(이 정보는 최신 웹 검색 결과를 기반으로 생성되었습니다.)"
                source_section += "\n참고 자료"
                
                for i, source in enumerate(self.last_web_sources[:5], 1):  # 상위 5개 소스만 표시
                    source_section += f"\n{source.get('title', f'출처 {i}')}"
                    if source.get('url'):
                        source_section += f"\n{source.get('url')}"
                
                response += source_section
            
            return response
        
        except Exception as e:
            logger.error(f"쿼리 처리 중 예외 발생: {e}", exc_info=True)
            return "죄송합니다. 요청을 처리하는 동안 오류가 발생했습니다: " + str(e)

    def generate_with_gemini(self, prompt: str) -> str:
        """Gemini 모델을 사용하여 응답 생성"""
        try:
            if not self.gemini_model:
                raise ValueError("Gemini 모델이 초기화되지 않았습니다.")
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                },
                safety_settings=safety_settings
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini 생성 중 오류: {e}", exc_info=True)
            raise

    def generate_with_medllama(self, prompt: str) -> str:
        """MedLlama 모델을 사용하여 응답 생성"""
        try:
            if not self.medllama_model:
                raise ValueError("MedLlama 모델이 초기화되지 않았습니다.")
            
            # MedLlama 모델 파라미터
            params = {
                "temperature": 0.1,
                "max_new_tokens": 2048,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
            
            # 추론 실행
            response = self.medllama_model.generate(prompt, **params)
            
            return response
        except Exception as e:
            logger.error(f"MedLlama 생성 중 오류: {e}", exc_info=True)
            raise