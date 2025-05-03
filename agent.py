# agent.py

import inspect
import json
import logging
import os
import re
import time
import asyncio  # asyncio 임포트 추가
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from dotenv import load_dotenv
from rag.mock_retriever import Document  # Document 클래스 import 추가

# --- 초기 설정 ---
# .env 파일 로드 (프로젝트 루트에 있다고 가정)
load_dotenv()

# 로깅 레벨 설정 (환경 변수 또는 기본값 INFO 사용)
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# (선택) xmltodict 설치 시 사용 가능
# try:
#     import xmltodict
#     xmltodict_available = True
# except ImportError:
#     xmltodict_available = False

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
    from rag.web_retriever import WebRetriever
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

    def __init__(self):
        '''에이전트 초기화 및 LLM 클라이언트 설정'''
        logger.info('MedicalAgent 초기화 시작...')

        # 설정값 인스턴스 변수로 저장
        self.analyzer_endpoint_base = AI_ANALYZER_API_ENDPOINT
        self.analyzer_api_key = AI_ANALYZER_API_KEY
        self.llm_provider = LLM_PROVIDER
        self.llm_api_key = LLM_API_KEY
        self.gemini_model_name = GEMINI_MODEL_NAME
        self.medllama_model_name = MEDLLAMA_MODEL_NAME
        self.medllama_model_path = MEDLLAMA_MODEL_PATH
        self.medllama_device = MEDLLAMA_DEVICE
        self.api_timeout = DEFAULT_API_TIMEOUT
        self.api_retries = DEFAULT_API_RETRIES
        self.api_retry_delay = DEFAULT_API_RETRY_DELAY

        # API 클라이언트 초기화
        self.db_initialized = False
        self.postgres_db = None
        self.pubmed_client = None
        self.kaggle_client = None

        # API 클라이언트 초기화 시도
        try:
            # PostgreSQL 데이터베이스 초기화
            from db.postgres_db import PostgresDB
            self.postgres_db = PostgresDB()
            logger.info('PostgreSQL 데이터베이스 클라이언트 초기화 완료')

            # PubMed API 클라이언트 초기화
            try:
                from api.pubmed_api import PubMedClient
                self.pubmed_client = PubMedClient()
                logger.info('PubMed API 클라이언트 초기화 완료')
            except ImportError:
                logger.warning('PubMed API 모듈을 가져올 수 없습니다. PubMed 관련 기능이 제한됩니다.')

            # Kaggle API 클라이언트 초기화
            try:
                from api.kaggle_api import KaggleClient
                self.kaggle_client = KaggleClient()
                logger.info('Kaggle API 클라이언트 초기화 완료')
            except ImportError:
                logger.warning('Kaggle API 모듈을 가져올 수 없습니다. Kaggle 관련 기능이 제한됩니다.')

            self.db_initialized = True

        except ImportError as e:
            logger.warning('데이터베이스/API 모듈을 가져올 수 없습니다: %s', e)
            logger.warning('기본 Firebase 데이터베이스를 사용합니다.')

        # LLM 클라이언트 초기화
        self.llm_clients = {}
        self.initialize_llm_client()
        self.llm_available = len(self.llm_clients) > 0

        # 사용 가능한 도구 목록 (LLM이 계획 수립 시 참고)
        # 설명에는 각 도구가 어떤 백엔드 API를 호출하는지 명시하는 것이 좋음
        self.available_tools = {
            'call_analyzer_generic': {
                'description': 'AISecurityAnalyzer 백엔드의 일반 분석 API(/analyze)를 호출하여 제공된 건강 데이터를 분석합니다.',
                'backend_path': '/analyze',
                'params': {'query_data': '분석할 건강 데이터 (JSON 객체)'}
            },
            'search_pubmed': {
                'description': 'PubMed API를 호출하여 특정 쿼리로 의학 논문을 검색합니다.',
                'backend_path': '/pubmed/search',
                'params': {'query': '검색어 (문자열)', 'max_results': '최대 결과 수 (정수, 기본값 5)'}
            },
            'search_kaggle': {
                'description': 'Kaggle API를 호출하여 특정 쿼리로 관련 데이터셋을 검색합니다.',
                'backend_path': '/kaggle/search',
                'params': {'query': '검색어 (문자열)', 'max_results': '최대 결과 수 (정수, 기본값 3)'}
            },
            'query_user_health_data': {
                'description': '환자 데이터베이스 API를 호출하여 특정 사용자의 건강 데이터를 조회합니다.',
                'backend_path': '/user/data',
                'params': {'user_id': '조회할 사용자 ID (문자열)'}
            },
            'semantic_search_documents': {
                'description': '의미론적 검색 API를 호출하여 내부 문서 코퍼스에서 관련된 정보를 검색합니다.',
                'backend_path': '/semantic/search',
                'params': {'query': '검색어 (문자열)', 'corpus_id': '검색 대상 코퍼스 ID (문자열, 기본값 \'medical_docs\')', 'top_k': '반환할 결과 수 (정수, 기본값 3)'}
            },
            # 필요시 백엔드에 구현된 다른 도구 API 추가
            # 'advanced_ai_analysis': {
            #     'description': 'AISecurityAnalyzer 백엔드의 고급 분석 API(/advanced/analyze)를 호출합니다.',
            #     'backend_path': '/advanced/analyze',
            #     'params': {'data_to_analyze': '분석할 데이터'}
            # }
        }
        logger.info('사용 가능한 도구: %s', list(self.available_tools.keys()))
        logger.info('MedicalAgent 초기화 완료.')

        # 웹 검색 기능 초기화
        self.web_search_enabled = web_search_available and os.getenv('ENABLE_WEB_SEARCH', 'false').lower() in ['true', '1', 'yes']
        self.web_retriever = None
        self.web_searcher = None
        self.web_scraper = None
        
        if self.web_search_enabled:
            try:
                # 웹 검색 초기화
                self.web_searcher = MedicalWebSearch(
                    google_api_key=os.getenv('GOOGLE_API_KEY'),
                    google_cse_id=os.getenv('GOOGLE_CSE_ID')
                )
                
                # 웹 스크래퍼 초기화
                self.web_scraper = MedicalWebScraper()
                
                # 웹 RAG 검색기 초기화
                self.web_retriever = WebRetriever()
                
                logger.info('웹 검색 기능 초기화 완료')
            except Exception as e:
                logger.error(f'웹 검색 초기화 오류: {e}', exc_info=True)
                self.web_search_enabled = False

    def initialize_llm_client(self) -> bool:
        '''LLM 클라이언트 초기화

        Returns:
            초기화 성공 여부
        '''
        # LLM 사용 환경 설정
        self.llm_provider = os.getenv('LLM_PROVIDER', 'gemini').lower()  # 'gemini', 'medllama', 'hybrid'
        self.primary_llm = os.getenv('LLM_PRIMARY', self.llm_provider).lower()

        self.llm_clients = {}
        self.llm_available = False

        logger.info('LLM 제공자 구성: %s (기본 모델: %s)', self.llm_provider, self.primary_llm)

        # API 키 로드
        self.llm_api_key = os.getenv('LLM_API_KEY', '')

        # MedLLaMA 설정
        if medllama_available:
            self.medllama_model_name = os.getenv('MEDLLAMA_MODEL_NAME', 'medllama-3-8b')
            self.medllama_model_path = os.getenv('MEDLLAMA_MODEL_PATH', './models/medllama')
            self.medllama_device = os.getenv('MEDLLAMA_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            # MedLLaMA 사용 불가능한 경우 LLM 제공자를 gemini로 강제 변경
            if self.llm_provider == 'medllama':
                logger.warning('MedLLaMA 라이브러리가 설치되지 않아 LLM 제공자를 \'gemini\'로 변경합니다.')
                self.llm_provider = 'gemini'
                self.primary_llm = 'gemini'
            elif self.llm_provider == 'hybrid':
                logger.warning('MedLLaMA 라이브러리가 설치되지 않아 하이브리드 모드에서 Gemini만 사용합니다.')

        # 재시도 설정
        self.api_retries = int(os.getenv('LLM_API_RETRIES', '2'))
        self.api_retry_delay = float(os.getenv('LLM_API_RETRY_DELAY', '1.5'))

        # LLM 클라이언트 초기화
        gemini_initialized = False
        medllama_initialized = False

        # Gemini 모델 초기화 (gemini 또는 hybrid 모드)
        if self.llm_provider in ['gemini', 'hybrid']:
            gemini_initialized = self._initialize_gemini_client()

        # MedLLaMA 모델 초기화 (medllama 또는 hybrid 모드)
        if medllama_available and self.llm_provider in ['medllama', 'hybrid']:
            medllama_initialized = self._initialize_medllama_client()

        # 초기화 결과 확인
        # hybrid 모드에서는 둘 중 하나라도 초기화되면 성공으로 간주
        if self.llm_provider == 'hybrid':
            self.llm_available = gemini_initialized or medllama_initialized
        # 다른 모드에서는 해당 모델이 초기화되어야 함
        elif self.llm_provider == 'gemini':
            self.llm_available = gemini_initialized
        elif self.llm_provider == 'medllama':
            self.llm_available = medllama_initialized

        if not self.llm_available:
            logger.warning('LLM 클라이언트 초기화 실패 (API 키 및 모델 경로 확인 필요)')
        else:
            logger.info('LLM 클라이언트 초기화 완료. 사용 가능한 모델: %s', ', '.join(self.llm_clients.keys()))

            # 기본 LLM이 초기화되지 않은 경우 대체 모델로 설정
            if self.primary_llm not in self.llm_clients:
                # 사용 가능한 첫 번째 모델을 기본값으로 설정
                if self.llm_clients:
                    self.primary_llm = list(self.llm_clients.keys())[0]
                    logger.warning('기본 LLM이 초기화되지 않았습니다. 대체 모델로 변경: %s', self.primary_llm)

        return self.llm_available

    def _initialize_gemini_client(self) -> bool:
        '''Gemini API 클라이언트 초기화

        Returns:
            초기화 성공 여부
        '''
        try:
            # API 키 확인
            if not self.llm_api_key:
                logger.warning('Gemini API 키가 제공되지 않았습니다. LLM_API_KEY 환경 변수를 설정하세요.')
                return False

            # Gemini 모델 설정
            self.gemini_model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-pro-preview-03-25')

            # Gemini API 클라이언트 초기화
            genai.configure(api_key=self.llm_api_key)

            # Gemini 모델 생성
            model = genai.GenerativeModel(self.gemini_model_name)
            self.llm_clients['gemini'] = model
            logger.info('Gemini API 클라이언트 초기화 완료 (모델: %s)', self.gemini_model_name)

            return True

        except Exception as e:
            logger.error(f'Gemini API 클라이언트 초기화 실패: {e}', exc_info=True)
            return False

    def _initialize_medllama_client(self) -> bool:
        '''의료 LLM 모델 초기화 (MedLLaMA 또는 대체 모델)

        Returns:
            초기화 성공 여부
        '''
        try:
            # 모델 경로 및 이름 설정
            model_name = self.medllama_model_name
            model_path = self.medllama_model_path

            # 모델 경로가 지정되지 않았거나 존재하지 않는 경우 공개 의료 LLM 사용
            if not model_path or not os.path.exists(model_path):
                logger.warning('로컬 MedLLaMA 모델 경로를 찾을 수 없습니다: %s', model_path)
                logger.info('공개 의료 LLM 모델을 사용합니다.')

                # 공개 의료 LLM 모델 목록 (다양한 크기와 특성의 모델)
                public_models = [
                    'epfl-llm/meditron-7b',           # 의료 특화 7B 모델
                    'epfl-llm/meditron-70b',          # 의료 특화 70B 모델 (더 강력)
                    'StanfordAIMI/stanford-deidentified-biomedical-llm', # 생물의학 LLM
                    'MaziyarPanahi/GatortronS',       # 의료 특화 LLM
                    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', # PubMed 특화 BERT
                    'medicalai/ClinicalBERT',         # 임상 특화 BERT
                    'allenai/scibert_scivocab_uncased', # 과학 문헌 특화 BERT
                    'facebook/galactica-1.3b',        # 과학 특화 LLM
                    'google/flan-t5-large'            # 일반 목적 모델 (대안)
                ]

                # 각 모델 시도
                model_loaded = False
                for public_model in public_models:
                    try:
                        logger.info('모델 시도: %s', public_model)
                        model_path = public_model
                        model_name = os.path.basename(public_model)

                        # 모델 크기 추정 (이름에 숫자가 있는 경우)
                        model_size_info = ''
                        if '70b' in public_model.lower():
                            model_size_info = ' (70B 대형 모델, 충분한 메모리 필요)'
                        elif '7b' in public_model.lower() or '7-b' in public_model.lower():
                            model_size_info = ' (7B 모델)'
                        elif '1.3b' in public_model.lower():
                            model_size_info = ' (1.3B 모델)'

                        logger.info('모델 로드 시도: %s%s', public_model, model_size_info)
                        model_loaded = True
                        break
                    except Exception as e:
                        logger.warning('모델 %s 접근 실패: %s', public_model, e)
                        continue

                if not model_loaded:
                    logger.error('사용 가능한 의료 LLM 모델을 찾을 수 없습니다.')
                    return False

            # 모델 로드를 위한 import
            from transformers import AutoConfig
            from transformers import AutoModelForCausalLM
            from transformers import AutoTokenizer

            # 로그 설정
            logger.info('의료 LLM 모델 로드 중 (%s) - 경로: %s', model_name, model_path)
            logger.info('사용 장치: %s', self.medllama_device)

            # 모델 타입 파악 (URL이나 경로에서 추론)
            model_type = 'causal_lm'  # 기본값
            if 'bert' in model_path.lower():
                model_type = 'bert'
            elif 't5' in model_path.lower():
                model_type = 't5'
            elif 'gpt' in model_path.lower():
                model_type = 'causal_lm'

            # 토크나이저 옵션
            tokenizer_options = {
                'trust_remote_code': True,
                'padding_side': 'left',  # 패딩 방향 설정
                'use_fast': True,  # 고속 토크나이저 사용
            }

            # 모델 옵션
            model_options = {
                'trust_remote_code': True,
                'device_map': 'auto' if self.medllama_device == 'cuda' else None,
                'torch_dtype': torch.float16 if self.medllama_device == 'cuda' else torch.float32,
                'low_cpu_mem_usage': True,  # 메모리 효율 향상
            }

            # 양자화 옵션 (메모리 절약)
            use_quantization = os.getenv('MEDLLAMA_USE_QUANTIZATION', 'True').lower() == 'true'

            if use_quantization and self.medllama_device == 'cuda':
                try:
                    from transformers import BitsAndBytesConfig

                    model_options['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_use_double_quant=True
                    )
                    logger.info('4비트 양자화 사용 활성화')
                except ImportError:
                    logger.warning('bitsandbytes 라이브러리가 없어 양자화를 사용할 수 없습니다.')

            # 모델 토큰 제한 적용
            max_model_length = 2048
            model_options['max_length'] = max_model_length

            # 모델 구성 확인 (선택적)
            try:
                config = AutoConfig.from_pretrained(model_path)
                if hasattr(config, 'max_length'):
                    max_model_length = config.max_length
                    model_options['max_length'] = max_model_length
                # 모델 타입 확인
                if hasattr(config, 'model_type'):
                    model_type = config.model_type
                logger.info('모델 최대 컨텍스트 길이: %s, 모델 타입: %s', max_model_length, model_type)
            except Exception as e:
                logger.warning('모델 구성 로드 중 오류 (무시): %s', e)

            # 토크나이저 로드
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_options)

                # 특수 토큰 설정
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                logger.error(f'토크나이저 로드 실패: {e}', exc_info=True)

                # 대체 토크나이저 시도
                try:
                    logger.info('대체 토크나이저 시도 중...')
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m')
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                except Exception as fallback_err:
                    logger.error('대체 토크나이저도 실패: %s', fallback_err)
                    return False

            # 모델 타입에 따른 로딩 방식 결정
            try:
                if model_type == 'bert':
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(model_path, **model_options)
                elif model_type == 't5':
                    from transformers import T5ForConditionalGeneration
                    model = T5ForConditionalGeneration.from_pretrained(model_path, **model_options)
                else:  # causal_lm 기본값
                    model = AutoModelForCausalLM.from_pretrained(model_path, **model_options)

                # 모델 성능 최적화
                if self.medllama_device == 'cuda':
                    model.eval()  # 평가 모드로 설정 (추론에 최적화)

                # 메모리 최적화
                if self.medllama_device == 'cpu':
                    # CPU에서 메모리 사용량 최적화
                    model = model.to('cpu', torch.float32)
            except Exception as e:
                logger.error(f'모델 로드 실패: {e}', exc_info=True)

                # 대체 모델 시도
                try:
                    logger.info('더 작은 대체 모델 시도 중...')
                    model_path = 'google/flan-t5-base'  # 더 작은 대체 모델
                    model_type = 't5'

                    from transformers import AutoTokenizer
                    from transformers import T5ForConditionalGeneration
                    model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    max_model_length = 512  # 작은 모델의 제한된 컨텍스트 길이

                    logger.info('대체 T5 모델 로드 성공')
                except Exception as fallback_err:
                    logger.error('대체 모델 로드도 실패: %s', fallback_err)
                    return False

            # 모델 및 토크나이저 저장
            self.llm_clients['medllama'] = {
                'model': model,
                'tokenizer': tokenizer,
                'model_name': model_name,
                'model_type': model_type,
                'max_length': max_model_length
            }

            # 모델 정보 기록
            if hasattr(model, 'config'):
                model_type = getattr(model.config, 'model_type', 'unknown')
                params_count = sum(p.numel() for p in model.parameters())
                logger.info('의료 LLM 모델 초기화 완료: %s, 유형: %s, 매개변수: %s',
                    model_name, model_type, f'{params_count:,}')
            else:
                logger.info('의료 LLM 모델 초기화 완료: %s', model_name)

            return True

        except Exception as e:
            logger.error(f'의료 LLM 모델 초기화 실패: {e}', exc_info=True)
            return False

    # --- Helper Function for Backend API Calls ---

    def _call_backend_api(self, method: str, path: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict[str, Any]:
        '''백엔드 API 호출 (PostgreSQL, PubMed, Kaggle API 사용)

        Args:
            method: HTTP 메서드 (get, post 등)
            path: API 경로
            params: 쿼리 파라미터
            data: 요청 본문 데이터

        Returns:
            API 응답 데이터
        '''
        logger.info('백엔드 API 호출: %s %s', method.upper(), path)

        try:
            # 각 API 경로에 따라 다른 처리
            if path == '/pubmed/search':
                # PubMed 검색 API
                query = params.get('query', '') if params else ''
                max_results = params.get('max_results', 5) if params else 5

                if not query:
                    return {'status': 'error', 'error': '검색어가 필요합니다.'}

                # PubMed API 클라이언트 확인
                if self.pubmed_client:
                    # PubMed API 직접 호출
                    articles = self.pubmed_client.search_articles(query, max_results=max_results)
                    return {'status': 'success', 'results': articles}
                else:
                    # PostgreSQL 데이터베이스에서 의학 문헌 검색
                    if self.db_initialized and self.postgres_db:
                        import asyncio
                        literature_results = asyncio.run(self.postgres_db.search_medical_literature(query, max_results))

                        results = []
                        for doc in literature_results:
                            results.append({
                                'id': doc.get('id', ''),
                                'title': doc.get('title', ''),
                                'authors': doc.get('authors', []),
                                'journal': doc.get('journal', ''),
                                'pubDate': doc.get('publication_date', ''),
                                'url': doc.get('doi', ''),
                                'abstract': doc.get('abstract', '')
                            })

                        return {'status': 'success', 'results': results}
                    else:
                        # 모의 데이터 반환
                        logger.warning('PubMed API 및 PostgreSQL을 사용할 수 없습니다. 모의 데이터 반환.')
                        return {
                            'status': 'success_mock',
                            'results': [
                                {
                                    'id': 'mock_pubmed_1',
                                    'title': f'모의 논문: {query}에 관한 연구',
                                    'authors': ['Mock Author 1', 'Mock Author 2'],
                                    'journal': 'Journal of Mock Medicine',
                                    'pubDate': '2024',
                                    'url': 'https://example.com/mock_doi',
                                    'abstract': f'{query}에 관한 모의 초록입니다. 이 데이터는 실제 PubMed 검색 결과가 아닙니다.'
                                }
                            ]
                        }

            elif path == '/kaggle/search':
                # Kaggle 검색 API
                query = params.get('query', '') if params else ''
                max_results = params.get('max_results', 3) if params else 3

                if not query:
                    return {'status': 'error', 'error': '검색어가 필요합니다.'}

                # Kaggle API 클라이언트 확인
                if self.kaggle_client:
                    # Kaggle API 직접 호출
                    datasets = self.kaggle_client.search_datasets(query, max_results=max_results)
                    return {'status': 'success', 'results': datasets}
                else:
                    # 모의 데이터 반환
                    logger.warning('Kaggle API를 사용할 수 없습니다. 모의 데이터 반환.')
                    return {
                        'status': 'success_mock',
                        'results': [
                            {
                                'id': 'mock_kaggle_1',
                                'title': f'모의 데이터셋: {query} 데이터',
                                'url': 'https://www.kaggle.com/datasets/mock/dataset1',
                                'description': f'{query}에 관한 모의 데이터셋입니다. 이 데이터는 실제 Kaggle 검색 결과가 아닙니다.'
                            }
                        ]
                    }

            elif path == '/semantic/search':
                # 의미론적 검색 API
                query = params.get('query', '') if params else ''
                corpus_id = params.get('corpus_id', 'medical_docs') if params else 'medical_docs'
                top_k = params.get('top_k', 3) if params else 3

                if not query:
                    return {'status': 'error', 'error': '검색어가 필요합니다.'}

                # PostgreSQL 데이터베이스 확인
                if self.db_initialized and self.postgres_db:
                    import asyncio

                    # 컬렉션 선택
                    if corpus_id == 'clinical_guidelines':
                        # 임상 가이드라인 검색
                        coll_name = 'clinical_guidelines'
                    else:
                        # 의학 문헌 검색 (기본값)
                        coll_name = 'medical_literature'

                    # 데이터베이스 쿼리
                    doc_results = asyncio.run(self.postgres_db.query_documents(coll_name, limit=top_k))

                    # 검색어와 관련된 문서만 필터링
                    query_lower = query.lower()
                    filtered_results = []

                    for doc in doc_results:
                        title = doc.get('title', '').lower()
                        content = doc.get('content', '').lower()
                        abstract = doc.get('abstract', '').lower()

                        if query_lower in title or query_lower in content or query_lower in abstract:
                            filtered_results.append({
                                'id': doc.get('id', ''),
                                'title': doc.get('title', ''),
                                'content': doc.get('content', '') or doc.get('abstract', ''),
                                'score': 0.85,  # 모의 점수
                                'metadata': {}
                            })

                    return {
                        'query': query,
                        'corpus_id': corpus_id,
                        'results': filtered_results[:top_k]
                    }
                else:
                    # 모의 데이터 반환
                    logger.warning('PostgreSQL을 사용할 수 없습니다. 모의 데이터 반환.')
                    return {
                        'query': query,
                        'corpus_id': corpus_id,
                        'results': [
                            {
                                'id': 'mock_doc_1',
                                'title': f'모의 문서: {query}에 관한 가이드라인',
                                'content': f'{query}에 관한 모의 내용입니다. 이 데이터는 실제 검색 결과가 아닙니다.',
                                'score': 0.9,
                                'metadata': {}
                            }
                        ]
                    }

            elif path == '/user/data':
                # 사용자 데이터 조회 API
                user_id = params.get('user_id', '') if params else ''

                if not user_id:
                    return {'status': 'error', 'error': '사용자 ID가 필요합니다.'}

                # PostgreSQL 데이터베이스 확인
                if self.db_initialized and self.postgres_db:
                    import asyncio

                    # 환자 정보 조회
                    patient_data = asyncio.run(self.postgres_db.get_patient_record(user_id))

                    if patient_data:
                        return patient_data

                # 모의 데이터 반환
                logger.warning('환자 ID \'%s\'의 데이터를 데이터베이스에서 찾을 수 없습니다. 모의 데이터 사용.', user_id)
                return {
                    'user_id': user_id,
                    'personal_info': {
                        'age': 45,
                        'gender': 'male',
                        'height': 175,
                        'weight': 78
                    },
                    'vitals': {
                        'heart_rate': 72,
                        'blood_pressure': {
                            'systolic': 125,
                            'diastolic': 82
                        },
                        'oxygen_level': 98,
                        'temperature': 36.5
                    },
                    'medical_history': {
                        'conditions': ['고혈압', '제2형 당뇨병'],
                        'allergies': ['페니실린'],
                        'surgeries': ['충수돌기염 수술 (2010)'],
                        'family_history': ['부: 심장질환', '모: 고혈압']
                    },
                    'medications': [
                        {
                            'name': '리시노프릴',
                            'dosage': '10mg',
                            'frequency': '1일 1회'
                        },
                        {
                            'name': '메트포르민',
                            'dosage': '500mg',
                            'frequency': '1일 2회'
                        }
                    ]
                }

            elif path == '/analyze':
                # 건강 데이터 분석 API
                if not data:
                    return {'status': 'error', 'error': '분석할 데이터가 필요합니다.'}

                # 간단한 모의 분석 수행
                analysis_result = {
                    'status': 'success',
                    'analysis_id': f'analysis_{int(time.time())}',
                    'results': {
                        'risk_factors': [],
                        'recommendations': [],
                        'confidence': 0.85
                    }
                }

                # 간단한 위험 요소 분석
                health_data = data.get('health_data', {})
                vitals = health_data.get('vitals', {})

                # 혈압 분석
                blood_pressure = vitals.get('blood_pressure', {})
                systolic = blood_pressure.get('systolic', 0)
                diastolic = blood_pressure.get('diastolic', 0)

                if systolic > 140 or diastolic > 90:
                    analysis_result['results']['risk_factors'].append({
                        'type': 'high_blood_pressure',
                        'severity': 'moderate' if systolic < 160 and diastolic < 100 else 'high',
                        'details': f'혈압이 {systolic}/{diastolic}로 정상 범위를 벗어났습니다.'
                    })

                    analysis_result['results']['recommendations'].append({
                        'type': 'lifestyle',
                        'action': '혈압 관리를 위해 저염식이, 규칙적인 운동, 체중 관리를 권장합니다.'
                    })

                # 당 수치 분석
                glucose = health_data.get('lab_results', {}).get('glucose', 0)
                if glucose > 126:
                    analysis_result['results']['risk_factors'].append({
                        'type': 'high_glucose',
                        'severity': 'moderate' if glucose < 180 else 'high',
                        'details': f'공복 혈당이 {glucose} mg/dL로 정상 범위를 벗어났습니다.'
                    })

                    analysis_result['results']['recommendations'].append({
                        'type': 'medical',
                        'action': '혈당 관리를 위한 의료 상담을 권장합니다.'
                    })

                return analysis_result

            else:
                # 지원되지 않는 API 경로
                logger.warning('지원되지 않는 API 경로: %s', path)
                return {'status': 'error', 'error': f'지원되지 않는 API 경로: {path}'}

        except Exception as e:
            logger.error(f'백엔드 API 호출 중 오류: {e}', exc_info=True)
            return {'status': 'error', 'error': f'백엔드 API 호출 오류: {str(e)}'}


    # --- 도구 함수 구현 (백엔드 호출 방식) ---

    def call_analyzer_generic(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        '''AISecurityAnalyzer 백엔드의 일반 분석 API 호출'''
        if not isinstance(query_data, dict):
            logger.error('call_analyzer_generic: \'query_data\'는 dictionary 타입이어야 합니다.')
            return {'error': 'Invalid parameter type: query_data must be a dictionary.', 'status': 'error_param_type'}
        tool_info = self.available_tools['call_analyzer_generic']
        return self._call_backend_api('post', tool_info['backend_path'], data=query_data)

    def search_pubmed(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        '''AISecurityAnalyzer 백엔드의 PubMed 검색 API 호출'''
        if not isinstance(query, str) or not query:
            logger.error('search_pubmed: \'query\'는 비어 있지 않은 문자열이어야 합니다.')
            return {'error': 'Invalid parameter: query must be a non-empty string.', 'status': 'error_param_value'}
        try:
            max_results = int(max_results)
            if max_results <= 0: max_results = 5
        except (ValueError, TypeError):
            logger.warning('search_pubmed: \'max_results\'가 유효한 정수가 아님 (%s). 기본값 5 사용.', max_results)
            max_results = 5

        tool_info = self.available_tools['search_pubmed']
        params = {'query': query, 'max_results': max_results}
        return self._call_backend_api('get', tool_info['backend_path'], params=params)

    def search_kaggle(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        '''AISecurityAnalyzer 백엔드의 Kaggle 검색 API 호출'''
        if not isinstance(query, str) or not query:
            logger.error('search_kaggle: \'query\'는 비어 있지 않은 문자열이어야 합니다.')
            return {'error': 'Invalid parameter: query must be a non-empty string.', 'status': 'error_param_value'}
        try:
            max_results = int(max_results)
            if max_results <= 0: max_results = 3
        except (ValueError, TypeError):
            logger.warning('search_kaggle: \'max_results\'가 유효한 정수가 아님 (%s). 기본값 3 사용.', max_results)
            max_results = 3

        tool_info = self.available_tools['search_kaggle']
        params = {'query': query, 'max_results': max_results}
        return self._call_backend_api('get', tool_info['backend_path'], params=params)

    def query_user_health_data(self, user_id: str) -> Dict[str, Any]:
        '''AISecurityAnalyzer 백엔드의 사용자 데이터 조회 API 호출'''
        if not isinstance(user_id, str) or not user_id:
            logger.error('query_user_health_data: \'user_id\'는 비어 있지 않은 문자열이어야 합니다.')
            return {'error': 'Invalid parameter: user_id must be a non-empty string.', 'status': 'error_param_value'}

        tool_info = self.available_tools['query_user_health_data']
        # 백엔드 API 경로에 user_id를 포함하거나 (예: /user/{user_id}/data), 파라미터로 전달
        # 아래는 파라미터로 전달하는 예시
        params = {'user_id': user_id}
        # path = tool_info['backend_path'].replace('{user_id}', user_id) # 경로에 포함 시
        return self._call_backend_api('get', tool_info['backend_path'], params=params)

    def semantic_search_documents(self, query: str, corpus_id: str = 'medical_docs', top_k: int = 3) -> Dict[str, Any]:
        '''AISecurityAnalyzer 백엔드의 의미론적 검색 API 호출'''
        if not isinstance(query, str) or not query:
            logger.error('semantic_search_documents: \'query\'는 비어 있지 않은 문자열이어야 합니다.')
            return {'error': 'Invalid parameter: query must be a non-empty string.', 'status': 'error_param_value'}
        if not isinstance(corpus_id, str): corpus_id = 'medical_docs'
        try:
            top_k = int(top_k)
            if top_k <= 0: top_k = 3
        except (ValueError, TypeError):
            logger.warning('semantic_search_documents: \'top_k\'가 유효한 정수가 아님 (%s). 기본값 3 사용.', top_k)
            top_k = 3

        tool_info = self.available_tools['semantic_search_documents']
        params = {'query': query, 'corpus_id': corpus_id, 'top_k': top_k}
        return self._call_backend_api('get', tool_info['backend_path'], params=params)


    # --- LLM 연동 함수 ---

    def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        '''대형 언어 모델 호출

        Args:
            prompt: 프롬프트 텍스트
            temperature: 응답 다양성 (0.0~1.0)
            max_tokens: 최대 토큰 수

        Returns:
            생성된 텍스트
        '''
        if not self.llm_available:
            return ''
            
        try:
            # 현재는 간단한 로컬 LLM 구현 (mock)
            # 실제 구현에서는 OpenAI, Anthropic, Google 등의 API를 사용
            if self.llm_config.get('type') == 'local':
                # 로컬 LLM 호출 (임시 구현)
                return self._mock_llm_response(prompt)
            
            elif self.llm_config.get('type') == 'openai':
                # OpenAI API 호출
                if 'openai' not in sys.modules:
                    import openai
                
                api_key = self.llm_config.get('api_key') or os.getenv('OPENAI_API_KEY')
                if not api_key:
                    logger.error('OpenAI API 키가 설정되지 않았습니다.')
                    return ''
                
                openai.api_key = api_key
                
                model_name = self.llm_config.get('model') or 'gpt-3.5-turbo'
                logger.info(f'OpenAI {model_name} 모델 호출')
                
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {'role': 'system', 'content': '당신은 의료 AI 에이전트입니다.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return response.choices[0].message.content.strip()
                
            elif self.llm_config.get('type') == 'anthropic':
                # Anthropic Claude API 호출 (필요시 구현)
                if 'anthropic' not in sys.modules:
                    import anthropic
                
                api_key = self.llm_config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    logger.error('Anthropic API 키가 설정되지 않았습니다.')
                    return ''
                
                client = anthropic.Anthropic(api_key=api_key)
                
                model_name = self.llm_config.get('model') or 'claude-2'
                logger.info(f'Anthropic {model_name} 모델 호출')
                
                response = client.completions.create(
                    model=model_name,
                    prompt=f'\n\nHuman: {prompt}\n\nAssistant:',
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature
                )
                
                return response.completion.strip()
                
            else:
                logger.error(f'지원되지 않는 LLM 유형: {self.llm_config.get(\'type\')}')
                return ''
                
        except Exception as e:
            logger.error(f'LLM 호출 중 오류 발생: {e}', exc_info=True)
            return ''
    
    def _mock_llm_response(self, prompt: str) -> str:
        '''테스트용 모의 LLM 응답 생성

        Args:
            prompt: 프롬프트 텍스트

        Returns:
            생성된 텍스트
        '''
        # 프롬프트에서 질문 추출
        question_match = re.search(r'사용자 질문: (.*?)(\n|\Z)', prompt)
        question = question_match.group(1) if question_match else '알 수 없는 질문'
        
        # 질문 유형에 따른 모의 응답
        if '당뇨병' in question:
            return (
                '당뇨병은 혈액 속 포도당(혈당) 수치가 높아지는 만성 대사 질환입니다. '
                '최신 치료법으로는 생활습관 개선(식이요법, 운동), 약물 치료(경구용 혈당강하제, 인슐린), '
                '그리고 최근 주목받는 GLP-1 수용체 작용제와 SGLT-2 억제제가 있습니다. '
                '개인의 상태에 맞는 치료법을 선택하는 것이 중요합니다.\n\n'
                '면책 조항: 이 정보는 교육 목적으로만 제공되며 의학적 조언을 대체하지 않습니다. '
                '건강 관련 결정은 항상 의료 전문가와 상담하신 후 내려주세요.'
            )
        
        elif '고혈압' in question:
            return (
                '고혈압은 혈액이 혈관 벽에 가하는 압력이 지속적으로 높은 상태를 말합니다. '
                '일반적으로 140/90 mmHg 이상일 때 고혈압으로 진단합니다. '
                '치료법으로는 저염식, 운동, 체중 관리 등의 생활습관 개선과 약물 치료가 있습니다. '
                '약물 치료에는 이뇨제, ACE 억제제, ARB, 칼슘 채널 차단제 등이 사용됩니다.\n\n'
                '면책 조항: 이 정보는 교육 목적으로만 제공되며 의학적 조언을 대체하지 않습니다. '
                '건강 관련 결정은 항상 의료 전문가와 상담하신 후 내려주세요.'
            )
        
        else:
            return (
                f'{question}에 대한 답변입니다. 이 질문에 대한 정확한 정보를 제공하기 위해 '
                '의학 문헌과 최신 연구를 참고하는 것이 중요합니다. '
                '더 자세한 정보는 의료 전문가와 상담하시기 바랍니다.\n\n'
                '면책 조항: 이 정보는 교육 목적으로만 제공되며 의학적 조언을 대체하지 않습니다. '
                '건강 관련 결정은 항상 의료 전문가와 상담하신 후 내려주세요.'
            )

    async def direct_web_search(self, query: str, search_type: str = 'medical', num_results: int = 5) -> list:
        '''직접 웹 검색 수행 (Agent API용)

        Args:
            query: 검색 쿼리
            search_type: 검색 유형 (general, medical, journal)
            num_results: 결과 수

        Returns:
            검색 결과 목록
        '''
        if not self.web_search_enabled or not self.web_retriever:
            logger.warning('웹 검색이 비활성화되었거나 초기화되지 않았습니다.')
            return []
            
        try:
            # 웹 검색 수행
            results = await self.web_retriever.search_web(query, search_type=search_type, num_results=num_results)
            return results
        except Exception as e:
            logger.error(f'직접 웹 검색 중 오류 발생: {e}', exc_info=True)
            return []


# --- 에이전트 사용 예시 ---
if __name__ == '__main__':
    # 필요한 라이브러리: pip install requests python-dotenv google-generativeai
    agent = MedicalAgent()

    # 테스트 쿼리 (user_id는 예시)
    test_cases = [
        {'query': 'What are the latest treatment options for type 2 diabetes according to PubMed?', 'user_id': None},
        {'query': 'Analyze my health data for potential risks.', 'user_id': 'user_mock_001'},
        {'query': 'Find Kaggle datasets related to Alzheimer\'s disease.', 'user_id': None},
        {'query': 'Explain the mechanism of action for statins using available documents.', 'user_id': 'user_test_abc'},
        {'query': 'Give me a general overview of hypertension.', 'user_id': None},
        {'query': 'How is my heart rate?', 'user_id': 'user_mock_001'}, # 사용자 데이터 조회 필요
        {'query': 'Tell me about common cold.', 'user_id': None}, # 간단한 정보성 쿼리
        {'query': 'Search for documents about headache treatments.', 'user_id': None},
        {'query': 'What are my recent lab results?', 'user_id': None} # User ID가 필요한 쿼리 (ID 없이)
    ]

    for i, case in enumerate(test_cases):
        query = case['query']
        user_id = case['user_id']
        print(f'\n{\'=\'*30} 테스트 케이스 {i+1} (User: {user_id}) {\'=\'*30}')
        print(f'Query: {query}')
        print('-' * 70)
        response = agent.process_query(query, user_id=user_id)
        print(f'\n[Agent Response]:\n{response}')
        print('='*80)
