# agent.py

import os
import json
import requests
import logging
import time
import re
import inspect
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple

# --- 초기 설정 ---
# .env 파일 로드 (프로젝트 루트에 있다고 가정)
load_dotenv()

# 로깅 레벨 설정 (환경 변수 또는 기본값 INFO 사용)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM 및 API 클라이언트 라이브러리 로드 시도
try:
    import google.generativeai as genai
    google_ai_available = True
except ImportError:
    logger.warning("Gemini 라이브러리(google-generativeai)가 설치되지 않았습니다. LLM 기능이 제한됩니다.")
    google_ai_available = False

# (선택) xmltodict 설치 시 사용 가능
# try:
#     import xmltodict
#     xmltodict_available = True
# except ImportError:
#     xmltodict_available = False

# --- 환경 변수 및 설정 로드 ---
# AI Security Analyzer Backend
AI_ANALYZER_API_ENDPOINT = os.getenv("AI_ANALYZER_API_ENDPOINT", "http://localhost:8000") # 기본 URL
AI_ANALYZER_API_KEY = os.getenv("AI_ANALYZER_API_KEY")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_API_KEY = os.getenv("LLM_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro-preview-03-25")

# API Call Settings
DEFAULT_API_TIMEOUT = int(os.getenv("DEFAULT_API_TIMEOUT", "20")) # 초 단위
DEFAULT_API_RETRIES = int(os.getenv("DEFAULT_API_RETRIES", "2")) # 재시도 횟수
DEFAULT_API_RETRY_DELAY = int(os.getenv("DEFAULT_API_RETRY_DELAY", "2")) # 재시도 간격 (초)

# PubMed Settings (백엔드 호출 시 필요)
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY") # 백엔드 호출 시 사용될 수 있음

# --- Mock 데이터 (예시, 실제로는 잘 사용되지 않음) ---
MOCK_ANALYZER_RESPONSE = {"status": "success_mock", "analysis_id": "mock-12345", "results": {"risk_score": 0.75}}
MOCK_USER_HEALTH_DATA = {"user_id": "user_mock_001", "vitals": {"heart_rate": 72}}


class MedicalAgentError(Exception):
    """Medical Agent 관련 오류 처리를 위한 사용자 정의 예외"""
    pass


class MedicalAgent:
    """
    의료 정보를 수집, 분석하고 전문가 수준의 답변을 생성하는 AI 에이전트.
    LLM과 백엔드 API를 활용하여 다양한 도구를 오케스트레이션합니다.
    """

    def __init__(self):
        """에이전트 초기화 및 LLM 클라이언트 설정"""
        logger.info("MedicalAgent 초기화 시작...")

        # 설정값 인스턴스 변수로 저장
        self.analyzer_endpoint_base = AI_ANALYZER_API_ENDPOINT
        self.analyzer_api_key = AI_ANALYZER_API_KEY
        self.llm_provider = LLM_PROVIDER
        self.llm_api_key = LLM_API_KEY
        self.gemini_model_name = GEMINI_MODEL_NAME
        self.api_timeout = DEFAULT_API_TIMEOUT
        self.api_retries = DEFAULT_API_RETRIES
        self.api_retry_delay = DEFAULT_API_RETRY_DELAY

        # LLM 클라이언트 초기화
        self.llm_client = self._initialize_llm_client()
        self.llm_available = self.llm_client is not None

        # 사용 가능한 도구 목록 (LLM이 계획 수립 시 참고)
        # 설명에는 각 도구가 어떤 백엔드 API를 호출하는지 명시하는 것이 좋음
        self.available_tools = {
            "call_analyzer_generic": {
                "description": "AISecurityAnalyzer 백엔드의 일반 분석 API(/analyze)를 호출하여 제공된 건강 데이터를 분석합니다.",
                "backend_path": "/analyze",
                "params": {"query_data": "분석할 건강 데이터 (JSON 객체)"}
            },
            "search_pubmed": {
                "description": "AISecurityAnalyzer 백엔드의 PubMed 검색 API(/pubmed/search)를 호출하여 특정 쿼리로 의학 논문을 검색합니다.",
                "backend_path": "/pubmed/search",
                "params": {"query": "검색어 (문자열)", "max_results": "최대 결과 수 (정수, 기본값 5)"}
            },
            "search_kaggle": {
                "description": "AISecurityAnalyzer 백엔드의 Kaggle 검색 API(/kaggle/search)를 호출하여 특정 쿼리로 관련 데이터셋을 검색합니다.",
                "backend_path": "/kaggle/search",
                "params": {"query": "검색어 (문자열)", "max_results": "최대 결과 수 (정수, 기본값 3)"}
            },
            "query_user_health_data": {
                "description": "AISecurityAnalyzer 백엔드의 사용자 데이터 조회 API(/user/data)를 호출하여 특정 사용자의 건강 데이터를 내부 시스템에서 조회합니다.",
                "backend_path": "/user/data",
                "params": {"user_id": "조회할 사용자 ID (문자열)"}
            },
            "semantic_search_documents": {
                "description": "AISecurityAnalyzer 백엔드의 의미론적 검색 API(/semantic/search)를 호출하여 내부 문서 코퍼스에서 관련된 정보를 검색합니다.",
                "backend_path": "/semantic/search",
                "params": {"query": "검색어 (문자열)", "corpus_id": "검색 대상 코퍼스 ID (문자열, 기본값 'medical_docs')", "top_k": "반환할 결과 수 (정수, 기본값 3)"}
            },
            # 필요시 백엔드에 구현된 다른 도구 API 추가
            # "advanced_ai_analysis": {
            #     "description": "AISecurityAnalyzer 백엔드의 고급 분석 API(/advanced/analyze)를 호출합니다.",
            #     "backend_path": "/advanced/analyze",
            #     "params": {"data_to_analyze": "분석할 데이터"}
            # }
        }
        logger.info(f"사용 가능한 도구: {list(self.available_tools.keys())}")
        logger.info("MedicalAgent 초기화 완료.")

    def _initialize_llm_client(self):
        """선택된 LLM 제공자에 따라 클라이언트 초기화"""
        if self.llm_provider == "gemini":
            if not google_ai_available:
                logger.error("Gemini 라이브러리 로드 실패. LLM 클라이언트 초기화 불가.")
                return None
            if not self.llm_api_key:
                logger.error("LLM_API_KEY가 설정되지 않았습니다. LLM 클라이언트 초기화 불가.")
                return None
            try:
                # 새로운 Gemini API 사용 방식
                genai.configure(api_key=self.llm_api_key)
                
                # 모델 초기화
                model = genai.GenerativeModel(self.gemini_model_name)
                logger.info(f"Gemini 클라이언트 초기화 완료 (모델: {self.gemini_model_name}).")
                return model
            except Exception as e:
                logger.error(f"Gemini 클라이언트 초기화 중 오류 발생: {e}", exc_info=True)
                return None
        # elif self.llm_provider == "openai":
            # OpenAI 클라이언트 초기화 로직
            # pass
        else:
            logger.error(f"지원하지 않는 LLM Provider: {self.llm_provider}")
            return None

    # --- Helper Function for Backend API Calls ---

    def _call_backend_api(self, method: str, path: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict[str, Any]:
        """AISecurityAnalyzer 백엔드 API를 호출하는 공통 함수 (재시도 포함)"""
        
        # 백엔드 API 호출 대신 모의 데이터 반환
        if path == "/pubmed/search":
            # PubMed 검색 모의 응답
            query = params.get("query", "") if params else ""
            max_results = params.get("max_results", 5) if params else 5
            
            mock_pubmed_results = {
                "results": [
                    {
                        "id": "35672740",
                        "title": "Heart Rate Variability and Clinical Outcomes in Patients With Cardiovascular Disease: A Systematic Review and Meta-Analysis",
                        "authors": "Smith JB, Johnson KL, Park S, et al.",
                        "journal": "Journal of the American Heart Association",
                        "pubDate": "2022 Jun 1",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/35672740/",
                        "abstract": "Heart rate variability (HRV) has emerged as a significant predictor of clinical outcomes in patients with cardiovascular disease. This systematic review and meta-analysis examined the relationship between HRV metrics and cardiovascular outcomes."
                    },
                    {
                        "id": "36159323",
                        "title": "Artificial Intelligence in Cardiology: Present and Future Applications",
                        "authors": "Lee JH, Kim YS, Choi HW, et al.",
                        "journal": "Korean Circulation Journal",
                        "pubDate": "2023 Jan 15",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/36159323/",
                        "abstract": "Recent advances in artificial intelligence (AI) have revolutionized various fields of medicine, including cardiology. This review focuses on the current and future applications of AI in cardiovascular medicine, including diagnosis, risk prediction, and treatment optimization."
                    },
                    {
                        "id": "35984215",
                        "title": "Current Management of Type 2 Diabetes: A Comprehensive Review",
                        "authors": "Park JH, Kim MJ, Lee YS, et al.",
                        "journal": "Diabetes & Metabolism Journal",
                        "pubDate": "2022 Aug 10",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/35984215/",
                        "abstract": "This comprehensive review examines the current evidence-based approaches to the management of type 2 diabetes, including pharmacological treatments, lifestyle modifications, and emerging therapies."
                    }
                ]
            }
            
            return mock_pubmed_results
            
        elif path == "/semantic/search":
            # 의미론적 검색 모의 응답
            query = params.get("query", "") if params else ""
            corpus_id = params.get("corpus_id", "medical_docs") if params else "medical_docs"
            top_k = params.get("top_k", 3) if params else 3
            
            mock_semantic_results = {
                "query": query,
                "corpus_id": corpus_id,
                "results": [
                    {
                        "id": "doc1",
                        "title": "고혈압과 심장 질환의 연관성",
                        "content": "고혈압은 심장 질환의 주요 위험 요소입니다. 지속적으로 혈압이 높으면 심장과 혈관에 더 많은 부담이 가해지게 됩니다.",
                        "score": 0.92,
                        "metadata": {
                            "author": "Kim MD, Sang Ho",
                            "year": 2022,
                            "source": "Korean Journal of Cardiology"
                        }
                    },
                    {
                        "id": "doc2",
                        "title": "심근경색: 증상 및 치료",
                        "content": "심근경색의 주요 증상으로는 가슴 통증, 호흡 곤란, 발한, 현기증 등이 있습니다. 즉각적인 치료는 세 가지 주요 치료법인 혈전용해제, 관상동맥 중재시술, 관상동맥 우회술을 포함합니다.",
                        "score": 0.85,
                        "metadata": {
                            "author": "Park MD, Ji Hoon",
                            "year": 2023,
                            "source": "International Cardiac Research"
                        }
                    },
                    {
                        "id": "doc3",
                        "title": "심장 질환 예방을 위한 생활 습관",
                        "content": "건강한 식습관, 규칙적인 운동, 스트레스 관리, 금연은 심장 질환 예방에 효과적입니다. 지중해식 식단이 심장 건강에 도움이 된다는 연구 결과가 있습니다.",
                        "score": 0.78,
                        "metadata": {
                            "author": "Lee MD, Min Jung",
                            "year": 2021,
                            "source": "Preventive Medicine Journal"
                        }
                    }
                ]
            }
            
            return mock_semantic_results
            
        elif path == "/user/data":
            # 사용자 데이터 모의 응답
            user_id = params.get("user_id", "") if params else ""
            
            mock_user_data = {
                "user_id": user_id,
                "personal_info": {
                    "age": 45,
                    "gender": "male",
                    "height": 175,
                    "weight": 78
                },
                "vitals": {
                    "heart_rate": 72,
                    "blood_pressure": {
                        "systolic": 125,
                        "diastolic": 82
                    },
                    "oxygen_level": 98,
                    "temperature": 36.5
                },
                "medical_history": {
                    "conditions": ["Hypertension", "Type 2 Diabetes"],
                    "allergies": ["Penicillin"],
                    "surgeries": ["Appendectomy (2010)"],
                    "family_history": ["Father: Heart Disease", "Mother: Hypertension"]
                },
                "medications": [
                    {
                        "name": "Lisinopril",
                        "dosage": "10mg",
                        "frequency": "Once daily"
                    },
                    {
                        "name": "Metformin",
                        "dosage": "500mg",
                        "frequency": "Twice daily"
                    }
                ],
                "lifestyle": {
                    "smoking": "Former smoker (quit 5 years ago)",
                    "alcohol": "Occasional",
                    "exercise": "Moderate (3 times per week)",
                    "diet": "Balanced, low sodium"
                },
                "recent_measurements": [
                    {
                        "date": "2025-04-20",
                        "heart_rate": 75,
                        "blood_pressure": {
                            "systolic": 128,
                            "diastolic": 84
                        },
                        "oxygen_level": 97
                    },
                    {
                        "date": "2025-04-23",
                        "heart_rate": 72,
                        "blood_pressure": {
                            "systolic": 125,
                            "diastolic": 82
                        },
                        "oxygen_level": 98
                    }
                ],
                "risk_assessment": {
                    "cardiac_risk_score": 25,
                    "stroke_risk_score": 18,
                    "diabetes_complication_risk": "Moderate"
                }
            }
            
            return mock_user_data
        
        else:
            # 기타 경로에 대한 모의 응답
            logger.info(f"백엔드 API 호출 (모의 데이터): {method.upper()} {path}")
            return {"status": "success_mock", "message": f"모의 데이터: {path} 경로에 대한 응답이 성공적으로 처리되었습니다."}

        # 아래 코드는 실제 백엔드 API 호출 코드로, 모의 데이터 사용 시 실행되지 않습니다.
        """
        url = f"{self.analyzer_endpoint_base}{path}"
        headers = {"Content-Type": "application/json"}
        if self.analyzer_api_key:
            headers["X-API-Key"] = self.analyzer_api_key

        logger.debug(f"백엔드 API 호출 시도: {method.upper()} {url}, Params: {params}, Data: {data is not None}")

        last_exception = None
        for attempt in range(self.api_retries + 1):
            try:
                response = requests.request(
                    method,
                    url,
                    headers=headers,
                    params=params, # GET 요청용 파라미터
                    json=data,     # POST/PUT 요청용 바디
                    timeout=self.api_timeout
                )
                response.raise_for_status() # 4xx 또는 5xx 상태 코드 시 HTTPError 발생
                logger.info(f"백엔드 API 호출 성공 ({method.upper()} {path}). 상태 코드: {response.status_code}")
                # JSON 응답이 아닐 수도 있으므로 확인 후 파싱
                if 'application/json' in response.headers.get('Content-Type', ''):
                    return response.json()
                else:
                    # JSON이 아닌 경우, 성공했지만 내용은 텍스트로 반환
                    return {"status": "success_non_json", "content": response.text}

            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"백엔드 API 호출 시간 초과 ({method.upper()} {path}). 시도 {attempt + 1}/{self.api_retries + 1}")
            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = e.response.status_code
                error_details = e.response.text
                logger.error(f"백엔드 API HTTP 오류 ({method.upper()} {path}): {status_code} - {error_details}. 시도 {attempt + 1}/{self.api_retries + 1}")
                # 4xx 오류는 재시도하지 않음 (클라이언트 오류 가능성)
                if 400 <= status_code < 500:
                    error_status = f"error_http_{status_code}"
                    if status_code == 401: error_status = "error_unauthorized"
                    elif status_code == 403: error_status = "error_forbidden_invalid_key"
                    elif status_code == 404: error_status = "error_not_found"
                    return {"error": f"HTTP error: {status_code}", "details": error_details, "status": error_status}
                # 5xx 오류는 재시도
            except Exception as e:
                last_exception = e
                logger.error(f"백엔드 API 호출 중 예외 발생 ({method.upper()} {path}): {e}. 시도 {attempt + 1}/{self.api_retries + 1}")
            
            # 마지막 시도가 아니면 지연 후 재시도
            if attempt < self.api_retries:
                time.sleep(self.api_retry_delay)
        
        # 모든 재시도 실패 후
        logger.error(f"백엔드 API 호출 최종 실패 ({method.upper()} {path}). 모든 재시도 완료.")
        error_msg = str(last_exception) if last_exception else "알 수 없는 오류"
        return {"error": f"All retry attempts failed: {error_msg}", "status": "error_all_retries_failed"}
        """


    # --- 도구 함수 구현 (백엔드 호출 방식) ---

    def call_analyzer_generic(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """AISecurityAnalyzer 백엔드의 일반 분석 API 호출"""
        if not isinstance(query_data, dict):
            logger.error("call_analyzer_generic: 'query_data'는 dictionary 타입이어야 합니다.")
            return {"error": "Invalid parameter type: query_data must be a dictionary.", "status": "error_param_type"}
        tool_info = self.available_tools['call_analyzer_generic']
        return self._call_backend_api("post", tool_info['backend_path'], data=query_data)

    def search_pubmed(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """AISecurityAnalyzer 백엔드의 PubMed 검색 API 호출"""
        if not isinstance(query, str) or not query:
             logger.error("search_pubmed: 'query'는 비어 있지 않은 문자열이어야 합니다.")
             return {"error": "Invalid parameter: query must be a non-empty string.", "status": "error_param_value"}
        try:
            max_results = int(max_results)
            if max_results <= 0: max_results = 5
        except (ValueError, TypeError):
            logger.warning(f"search_pubmed: 'max_results'가 유효한 정수가 아님 ({max_results}). 기본값 5 사용.")
            max_results = 5

        tool_info = self.available_tools['search_pubmed']
        params = {"query": query, "max_results": max_results}
        return self._call_backend_api("get", tool_info['backend_path'], params=params)

    def search_kaggle(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """AISecurityAnalyzer 백엔드의 Kaggle 검색 API 호출"""
        if not isinstance(query, str) or not query:
             logger.error("search_kaggle: 'query'는 비어 있지 않은 문자열이어야 합니다.")
             return {"error": "Invalid parameter: query must be a non-empty string.", "status": "error_param_value"}
        try:
            max_results = int(max_results)
            if max_results <= 0: max_results = 3
        except (ValueError, TypeError):
            logger.warning(f"search_kaggle: 'max_results'가 유효한 정수가 아님 ({max_results}). 기본값 3 사용.")
            max_results = 3

        tool_info = self.available_tools['search_kaggle']
        params = {"query": query, "max_results": max_results}
        return self._call_backend_api("get", tool_info['backend_path'], params=params)

    def query_user_health_data(self, user_id: str) -> Dict[str, Any]:
        """AISecurityAnalyzer 백엔드의 사용자 데이터 조회 API 호출"""
        if not isinstance(user_id, str) or not user_id:
             logger.error("query_user_health_data: 'user_id'는 비어 있지 않은 문자열이어야 합니다.")
             return {"error": "Invalid parameter: user_id must be a non-empty string.", "status": "error_param_value"}

        tool_info = self.available_tools['query_user_health_data']
        # 백엔드 API 경로에 user_id를 포함하거나 (예: /user/{user_id}/data), 파라미터로 전달
        # 아래는 파라미터로 전달하는 예시
        params = {"user_id": user_id}
        # path = tool_info['backend_path'].replace("{user_id}", user_id) # 경로에 포함 시
        return self._call_backend_api("get", tool_info['backend_path'], params=params)

    def semantic_search_documents(self, query: str, corpus_id: str = "medical_docs", top_k: int = 3) -> Dict[str, Any]:
        """AISecurityAnalyzer 백엔드의 의미론적 검색 API 호출"""
        if not isinstance(query, str) or not query:
             logger.error("semantic_search_documents: 'query'는 비어 있지 않은 문자열이어야 합니다.")
             return {"error": "Invalid parameter: query must be a non-empty string.", "status": "error_param_value"}
        if not isinstance(corpus_id, str): corpus_id = "medical_docs"
        try:
            top_k = int(top_k)
            if top_k <= 0: top_k = 3
        except (ValueError, TypeError):
            logger.warning(f"semantic_search_documents: 'top_k'가 유효한 정수가 아님 ({top_k}). 기본값 3 사용.")
            top_k = 3

        tool_info = self.available_tools['semantic_search_documents']
        params = {"query": query, "corpus_id": corpus_id, "top_k": top_k}
        return self._call_backend_api("get", tool_info['backend_path'], params=params)


    # --- LLM 연동 함수 ---

    def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> Optional[str]:
        """LLM API를 호출하여 응답 생성 (재시도 포함)"""
        if not self.llm_available or self.llm_client is None:
            logger.warning("LLM 기능 사용 불가 (API 키 또는 클라이언트 문제).")
            return None

        logger.debug(f"LLM 호출 시작. Provider: {self.llm_provider}, Model: {self.gemini_model_name}")
        last_exception = None

        for attempt in range(self.api_retries + 1):
            try:
                if self.llm_provider == "gemini":
                    # Gemini 2.5 모델 API 호출
                    response = self.llm_client.generate_content(
                        prompt,
                        generation_config={
                            "temperature": temperature, 
                            "max_output_tokens": max_tokens,
                            "top_p": 0.95,
                            "top_k": 40
                        }
                    )
                    
                    # 응답 확인
                    if hasattr(response, 'text'):
                        logger.info("LLM 호출 성공.")
                        return response.text
                    else:
                        # 응답에 text 속성이 없는 경우
                        if hasattr(response, 'prompt_feedback'):
                            prompt_feedback = response.prompt_feedback
                            logger.warning(f"Gemini 응답 생성 실패 (콘텐츠 필터링 가능성). Feedback: {prompt_feedback}")
                            raise MedicalAgentError(f"LLM content generation filtered. Feedback: {prompt_feedback}")
                        else:
                            logger.error(f"Gemini 응답에 'text' 속성이 없습니다. 응답: {response}")
                            raise MedicalAgentError("LLM response missing 'text' attribute.")

                # elif self.llm_provider == "openai":
                    # OpenAI API 호출 로직
                    # pass
                else:
                    logger.error(f"지원하지 않는 LLM Provider: {self.llm_provider}")
                    return None # 지원하지 않는 제공자는 재시도 의미 없음

            except MedicalAgentError as e: # 콘텐츠 필터링 등 재시도 불필요 오류
                 last_exception = e
                 break # 재시도 중단
            except Exception as e:
                # API 오류, 네트워크 오류 등 재시도 가능 오류 처리
                last_exception = e
                logger.error(f"LLM API 호출 중 오류 발생 ({self.llm_provider}). 시도 {attempt + 1}/{self.api_retries + 1}: {e}", exc_info=True)

            # 재시도 전 대기
            if attempt < self.api_retries:
                logger.info(f"{self.api_retry_delay}초 후 LLM 호출 재시도...")
                time.sleep(self.api_retry_delay)

        # 모든 재시도 실패 시
        logger.error(f"LLM API 호출 최종 실패 후 {self.api_retries}회 재시도. 마지막 오류: {last_exception}")
        return None


    def _parse_llm_json_plan(self, llm_response: str) -> Optional[List[Dict[str, Any]]]:
        """LLM 응답에서 JSON 형식의 도구 계획을 파싱 (방어적)"""
        if not llm_response:
            return None
        # ```json ... ``` 코드 블록 추출 시도
        match = re.search(r"""```json\s*([\s\S]*?)\s*```""", llm_response, re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
            logger.debug("LLM 응답에서 JSON 코드 블록 추출 성공.")
        else:
            # 코드 블록이 없으면 전체 응답을 JSON으로 가정
            json_str = llm_response.strip()
            logger.debug("LLM 응답에 JSON 코드 블록 없음. 전체 응답을 JSON으로 가정.")

        try:
            plan = json.loads(json_str)
            if isinstance(plan, list):
                # 추가 검증: 리스트의 각 항목이 딕셔너리인지, 'tool' 키가 있는지 등
                if all(isinstance(item, dict) and 'tool' in item for item in plan):
                    logger.info("LLM 도구 계획 JSON 파싱 및 기본 검증 성공.")
                    return plan
                else:
                     logger.error("파싱된 JSON 계획의 구조가 올바르지 않습니다 (리스트 내 항목 오류).")
                     return None
            else:
                logger.error(f"파싱된 JSON 계획이 리스트 타입이 아닙니다 (타입: {type(plan)}).")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"LLM 도구 계획 JSON 파싱 실패: {e}. 원본 문자열(일부): {json_str[:200]}...")
            return None


    def _plan_tool_usage(self, query: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """LLM을 사용하여 사용자 쿼리 분석 및 도구 사용 계획 수립"""
        logger.info(f"LLM 기반 도구 사용 계획 수립 시작 (User: {user_id}): '{query}'")
        if not self.llm_available:
             logger.warning("LLM 사용 불가. 기본 계획 수립 로직 사용.")
             # LLM 없을 때 사용할 매우 기본적인 대체 로직 (상황에 맞게 수정 필요)
             plan = [{"tool": "search_pubmed", "params": {"query": query, "max_results": 2}}]
             if user_id:
                 plan.append({"tool": "query_user_health_data", "params": {"user_id": user_id}})
             return plan

        # 사용 가능한 도구 설명을 JSON 문자열로 준비 (LLM 프롬프트용)
        tools_for_prompt = {}
        for name, info in self.available_tools.items():
             tools_for_prompt[name] = {
                 "description": info["description"],
                 "parameters": info["params"] # 파라미터 정보도 LLM에게 제공
             }
        available_tools_json = json.dumps(tools_for_prompt, indent=2)

        # LLM에게 전달할 프롬프트 설계
        prompt = f"""You are an intelligent AI assistant planning tasks for a medical agent.
        Analyze the user query and determine the sequence of tools needed to gather the necessary information.
        User Query: "{query}"
        User ID: {user_id if user_id else 'Not provided'}

        Available Tools:
        ```json
        {available_tools_json}
        ```

        Instructions:
        1.  Consider the user query and User ID (if provided) to select the most relevant tools.
        2.  If the user asks about their own data ('my data', 'my health'), use the 'query_user_health_data' tool with the provided User ID. If User ID is not provided for such query, you MUST ask the user for their ID first (do not include the tool in the plan, return an empty list `[]` instead and I will handle asking the user).
        3.  Determine the necessary parameters for each selected tool based on the query.
        4.  Output your plan ONLY as a valid JSON list of dictionaries. Each dictionary must have 'tool' (the exact tool name string) and 'params' (a dictionary of parameters).
        5.  If no tools are needed or the query is too vague, output an empty JSON list `[]`.
        6.  Think step-by-step to construct the plan. Example: If the query asks for recent research and user data analysis, the plan might involve 'search_pubmed' first, then 'query_user_health_data', then potentially 'call_analyzer_generic'.

        Generate the JSON plan for the query: "{query}"
        JSON Plan:
        """
        logger.debug(f"도구 계획 LLM 프롬프트 (일부):\n{prompt[:1000]}...")

        raw_plan_response = self._call_llm(prompt, temperature=0.2, max_tokens=1024) # 계획은 낮은 temperature

        if not raw_plan_response:
            logger.error("LLM으로부터 도구 사용 계획을 받지 못했습니다.")
            raise MedicalAgentError("Failed to get tool usage plan from LLM.")

        logger.debug(f"LLM 원시 계획 응답: {raw_plan_response}")

        # LLM 응답 파싱 및 검증
        plan = self._parse_llm_json_plan(raw_plan_response)

        if plan is None:
             logger.error("LLM 도구 계획 파싱 또는 검증 실패.")
             raise MedicalAgentError("Failed to parse or validate the LLM tool plan response.")

        # 추가 검증: 계획된 도구가 실제로 존재하는지 확인
        valid_plan = []
        for step in plan:
            tool_name = step.get("tool")
            if tool_name in self.available_tools:
                 valid_plan.append(step)
            else:
                 logger.warning(f"LLM이 계획한 도구 '{tool_name}'는 사용 가능 목록에 없습니다. 이 단계를 무시합니다.")

        # User ID 누락 시 사용자 데이터 조회 도구 제거 (LLM 프롬프트 지침 강화)
        final_valid_plan = []
        for step in valid_plan:
            tool_name = step.get("tool")
            if tool_name == 'query_user_health_data' and not user_id:
                logger.warning("사용자 데이터 조회 계획이 있었으나 User ID가 제공되지 않아 해당 단계를 제거합니다.")
                continue
            final_valid_plan.append(step)

        logger.info(f"LLM이 수립하고 검증된 최종 도구 사용 계획: {final_valid_plan}")
        return final_valid_plan


    def _preprocess_data_for_synthesis(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 응답 생성 전, 수집된 데이터(특히 오류)를 정리/요약"""
        processed_data = {}
        for tool_name, result in collected_data.items():
            if isinstance(result, dict) and result.get("status", "").startswith("error"):
                # 오류 발생 시 간단한 메시지로 변환
                error_msg = result.get("error", "Unknown error")
                status = result.get("status", "error")
                details = result.get("details", "") # HTTP 오류 등 세부 정보
                summary = f"Failed ({status}): {error_msg}"
                if details and len(details) < 150: # 너무 길지 않은 세부 정보만 포함 (길이 조정)
                     summary += f" | Details: {details}"
                processed_data[tool_name] = {"status": "error", "summary": summary}
                logger.warning(f"도구 {tool_name} 실행 실패 요약: {summary}")
            elif isinstance(result, dict) and result.get("status") == "success_empty":
                 processed_data[tool_name] = {"status": "success", "summary": result.get("message", "No relevant information found.")}
            elif isinstance(result, dict):
                 # 성공 데이터도 너무 크면 요약 필요 (여기서는 status와 content 위주로 전달)
                 status = result.get("status", "success")
                 # 민감 정보나 불필요하게 큰 데이터 필터링/요약 로직 추가 가능
                 # 예: result_summary = {"status": status, "data_preview": str(result)[:500]+"..."}
                 processed_data[tool_name] = result # 일단 원본 전달, 필요시 위 요약 로직 적용
            else:
                 # 예상치 못한 타입의 결과
                 processed_data[tool_name] = {"status": "unknown", "summary": f"Unexpected result format: {str(result)[:100]}..."}
        return processed_data


    def _synthesize_response(self, query: str, user_id: Optional[str], processed_data: Dict[str, Any]) -> str:
        """LLM을 사용하여 정리된 정보 기반 최종 답변 생성"""
        logger.info("LLM 기반 답변 생성 시작 (정리된 데이터 사용)...")
        if not self.llm_available:
             logger.warning("LLM 사용 불가. 기본 응답 생성 로직 사용.")
             response_parts = [f"Query: {query}\nUser ID: {user_id}\nCollected Data Summary:"]
             for tool, data in processed_data.items():
                  status = data.get("status", "unknown")
                  summary = data.get("summary", str(data)[:200]+"...") if status != "success" else str(data)[:200]+"..."
                  response_parts.append(f"- {tool}: Status: {status}, Info: {summary}")
             response_parts.append("\nNote: This is a basic response as LLM is unavailable.")
             # 기본 응답에도 Disclaimer 추가
             response_parts.append("\n\n**Disclaimer:** I am an AI assistant. This information is for informational purposes only and does not constitute medical advice. Please consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment.")
             return "\n".join(response_parts)

        # LLM에게 전달할 프롬프트 설계
        prompt = f"""You are 'MediGuide AI', a world-class medical AI assistant embodying the persona of an expert, empathetic, and cautious medical professional.
        Your task is to synthesize the gathered information into a comprehensive, evidence-based response for the user.

        User Query: "{query}"
        User ID: {user_id if user_id else 'Not applicable'}

        Collected Information (Summarized - includes success and failure reports from tools):
        ```json
        {json.dumps(processed_data, indent=2, ensure_ascii=False)}
        ```

        Response Instructions:
        1.  **Acknowledge the Query:** Start by briefly acknowledging the user's query.
        2.  **Synthesize Findings:** Combine information from different tools logically. Highlight key findings, potential risks, relevant research, or dataset information based *only* on the `processed_data`.
        3.  **Cite Evidence:** When referring to specific tool results (like PubMed articles or analysis findings), implicitly or explicitly mention the source based on the keys in `processed_data` (e.g., "Recent studies found via PubMed search suggest..." or "The analysis of your provided data indicated...").
        4.  **Address Errors/Limitations:** If a tool failed (indicated by `status: "error"` in `processed_data`), acknowledge this limitation gracefully using the provided error `summary`. Example: "I encountered an issue while searching Kaggle datasets (Failed: API connection error)." or "Unfortunately, I was unable to retrieve relevant documents due to a technical difficulty." Do not expose raw technical details beyond the summary.
        5.  **Handle No Results:** If a tool succeeded but found no relevant information (`status: "success"` with an empty result or a specific message like "No relevant information found."), state this clearly. Example: "My search did not find recent PubMed articles specifically matching your query criteria."
        6.  **Maintain Persona:** Use clear, professional, and empathetic language. Avoid overly technical jargon unless necessary, and explain complex terms simply.
        7.  **Include Disclaimer:** ALWAYS conclude your response with the following exact disclaimer on a new line:
            **Disclaimer:** I am an AI assistant. This information is for informational purposes only and does not constitute medical advice. Please consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment.
        8.  **Structure:** Organize the response logically. Start with a summary, elaborate on findings, address limitations, and end with the disclaimer.
        9.  **Focus:** Base your response *only* on the provided query and the `processed_data`. Do not invent information or speculate beyond the data.

        Generate the final, user-facing response for MediGuide AI:
        """
        logger.debug(f"답변 종합 LLM 프롬프트 (일부):\n{prompt[:1000]}...")

        final_response = self._call_llm(prompt, temperature=0.6, max_tokens=3072) # 약간 낮은 temperature, 충분한 토큰

        if not final_response:
            logger.error("LLM으로부터 최종 답변을 생성하지 못했습니다.")
            # LLM 실패 시 대체 응답
            fallback_response = f"""I apologize, but I encountered an issue while generating the final response. I was able to gather some information, but could not synthesize it into a full answer. Please try rephrasing your query or contact support if the issue persists.\n\nCollected data summary:\n{json.dumps(processed_data, indent=2, ensure_ascii=False)}

**Disclaimer:** I am an AI assistant. This information is for informational purposes only and does not constitute medical advice. Please consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment."""
            return fallback_response

        # 간단한 후처리: 앞뒤 공백 제거, 여러 빈 줄 하나로 축소
        final_response = re.sub(r'\n\s*\n', '\n\n', final_response.strip())

        # Disclaimer가 포함되었는지 확인하고 없으면 추가 (LLM이 지침을 따르지 않을 경우 대비)
        disclaimer = "**Disclaimer:** I am an AI assistant. This information is for informational purposes only and does not constitute medical advice. Please consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment."
        if disclaimer not in final_response:
             logger.warning("LLM 응답에 Disclaimer가 누락되어 강제로 추가합니다.")
             final_response += "\n\n" + disclaimer

        logger.info("LLM 기반 답변 생성 완료.")
        return final_response


    def process_query(self, query: str, user_id: Optional[str] = None) -> str:
        """사용자 쿼리 처리의 전체 파이프라인"""
        logger.info(f"쿼리 처리 시작 (User: {user_id}): '{query}'")
        collected_data: Dict[str, Any] = {}

        try:
            # 1. 도구 사용 계획 수립 (LLM 사용)
            tool_plan = self._plan_tool_usage(query, user_id)

            # 1.5 사용자 ID 요청 처리 (LLM이 빈 계획을 반환했을 수 있음)
            if not tool_plan and "my " in query.lower() and user_id is None:
                 logger.info("LLM이 사용자 데이터 관련 쿼리에 대해 계획을 생성하지 않았습니다 (User ID 누락 가능성). 사용자에게 ID 요청 필요.")
                 return "To access your personal health data, please provide your User ID."

            # 2. 계획된 도구 실행 및 정보 수집
            if not tool_plan:
                 logger.warning("실행할 도구 계획이 없습니다. LLM이 계획을 생성하지 않았거나 쿼리가 모호할 수 있습니다.")
                 collected_data["planning_info"] = {"status": "no_plan", "summary": "No tool execution plan was generated. The query might be too general or require clarification."}
            else:
                for step in tool_plan:
                    tool_name = step.get("tool")
                    raw_params = step.get("params", {})

                    if not tool_name or tool_name not in self.available_tools:
                         logger.error(f"계획된 도구 '{tool_name}'가 유효하지 않습니다.")
                         collected_data[f"error_{tool_name or 'invalid_tool'}"] = {"status": "error_invalid_tool", "summary": f"Invalid tool '{tool_name}' specified in plan."}
                         continue

                    tool_method = getattr(self, tool_name, None)
                    if not callable(tool_method):
                         logger.error(f"도구 '{tool_name}'에 해당하는 실행 가능한 메서드가 없습니다.")
                         collected_data[f"error_{tool_name}"] = {"status": "error_tool_not_callable", "summary": f"Tool '{tool_name}' is not callable."}
                         continue

                    # 함수 시그니처 확인하여 필요한 파라미터만 전달
                    sig = inspect.signature(tool_method)
                    valid_params = {}
                    if isinstance(raw_params, dict):
                        for param_name, param in sig.parameters.items():
                            if param_name in raw_params:
                                # 타입 변환 시도 (예: max_results를 int로)
                                target_type = sig.parameters[param_name].annotation
                                raw_value = raw_params[param_name]
                                if target_type is int:
                                    try:
                                        valid_params[param_name] = int(raw_value)
                                    except (ValueError, TypeError):
                                        logger.warning(f"Parameter '{param_name}' for tool '{tool_name}' should be int, but got {raw_value}. Using default or skipping.")
                                        # 기본값 사용 또는 오류 처리 로직 추가 가능
                                elif target_type is str:
                                    valid_params[param_name] = str(raw_value)
                                # 다른 타입 (dict 등) 필요시 추가
                                else:
                                     valid_params[param_name] = raw_value # 타입 검사 없이 그대로 전달

                    else:
                         logger.warning(f"도구 '{tool_name}'의 파라미터가 딕셔너리 형태가 아닙니다: {raw_params}. 파라미터 없이 호출합니다.")


                    logger.info(f"도구 실행: {tool_name} (파라미터: {valid_params})")
                    try:
                        # user_id 자동 추가 (query_user_health_data의 경우)
                        # if tool_name == 'query_user_health_data' and 'user_id' not in valid_params and user_id:
                        #     valid_params['user_id'] = user_id
                        # --> LLM이 계획 시 user_id를 포함하도록 유도하는 것이 더 좋음

                        result = tool_method(**valid_params)
                        collected_data[tool_name] = result
                        logger.info(f"도구 {tool_name} 실행 완료. Status: {result.get('status', 'N/A')}")
                    except Exception as e:
                        # tool_method 실행 자체에서 발생한 예외 처리
                        logger.error(f"도구 {tool_name} 실행 중 예상치 못한 내부 오류 발생: {e}", exc_info=True)
                        collected_data[tool_name] = {"status": "error_tool_exception", "error": f"Unexpected error during tool execution: {e}", "summary": f"Failed to execute tool {tool_name} due to an internal error."}


            # 3. 수집된 정보 전처리
            processed_data = self._preprocess_data_for_synthesis(collected_data)

            # 4. 정리된 정보 바탕으로 답변 생성 (LLM 사용)
            final_response = self._synthesize_response(query, user_id, processed_data)

            logger.info("쿼리 처리 완료.")
            return final_response

        except MedicalAgentError as e:
             logger.error(f"MedicalAgent 처리 오류: {e}", exc_info=True)
             # 사용자에게 보여줄 오류 메시지 조정
             user_error_message = f"I encountered an issue while processing your request: {e}. Please try again later or rephrase your query."
             if "LLM content generation filtered" in str(e):
                  user_error_message = "I am sorry, but I cannot generate a response for this query due to safety guidelines. Please try a different query."
             return user_error_message + f"\n\n**Disclaimer:** I am an AI assistant. This information is for informational purposes only and does not constitute medical advice. Please consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment."

        except Exception as e:
             logger.error(f"쿼리 처리 중 예상치 못한 최상위 오류 발생: {e}", exc_info=True)
             return f"I apologize, but an unexpected system error occurred. Please try again later.\n\n**Disclaimer:** I am an AI assistant. This information is for informational purposes only and does not constitute medical advice. Please consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment."


# --- 에이전트 사용 예시 ---
if __name__ == "__main__":
    # 필요한 라이브러리: pip install requests python-dotenv google-generativeai
    agent = MedicalAgent()

    # 테스트 쿼리 (user_id는 예시)
    test_cases = [
        {"query": "What are the latest treatment options for type 2 diabetes according to PubMed?", "user_id": None},
        {"query": "Analyze my health data for potential risks.", "user_id": "user_mock_001"},
        {"query": "Find Kaggle datasets related to Alzheimer's disease.", "user_id": None},
        {"query": "Explain the mechanism of action for statins using available documents.", "user_id": "user_test_abc"},
        {"query": "Give me a general overview of hypertension.", "user_id": None},
        {"query": "How is my heart rate?", "user_id": "user_mock_001"}, # 사용자 데이터 조회 필요
        {"query": "Tell me about common cold.", "user_id": None}, # 간단한 정보성 쿼리
        {"query": "Search for documents about headache treatments.", "user_id": None},
        {"query": "What are my recent lab results?", "user_id": None} # User ID가 필요한 쿼리 (ID 없이)
    ]

    for i, case in enumerate(test_cases):
        query = case["query"]
        user_id = case["user_id"]
        print(f"\n{'='*30} 테스트 케이스 {i+1} (User: {user_id}) {'='*30}")
        print(f"Query: {query}")
        print("-" * 70)
        response = agent.process_query(query, user_id=user_id)
        print(f"\n[Agent Response]:\n{response}")
        print("="*80) 