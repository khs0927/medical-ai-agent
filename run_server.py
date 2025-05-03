#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
의료 AI 에이전트 서버 실행 스크립트

FastAPI 서버를 시작하고 필요한 경우 데이터베이스 초기화
"""
import argparse
from datetime import datetime
import json
import logging
import logging.handlers
import os
from pathlib import Path
import signal
import sys

from dotenv import load_dotenv
import uvicorn

# 로깅 설정
DEFAULT_LOG_LEVEL = os.getenv('LOG_LEVEL', 'info').lower()

# 로그 디렉토리 설정
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

# 로그 포맷 설정
log_formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 로그 핸들러 설정
def setup_logging(log_level):
    """로깅 시스템 설정"""
    level = getattr(logging, log_level.upper())

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # 파일 핸들러 (로그 순환)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / 'server.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)

    # 에러 전용 파일 핸들러
    error_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / 'error.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    error_handler.setFormatter(log_formatter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_handler)

    return root_logger

# 전역 로거 인스턴스
logger = None

# 환경 변수 로드
load_dotenv()

# 기본 설정값
DEFAULT_HOST = os.getenv('API_HOST', '0.0.0.0')
DEFAULT_PORT = int(os.getenv('API_PORT', '8000'))
DEFAULT_WORKERS = int(os.getenv('API_WORKERS', '1'))

# 데이터베이스 타입
DB_TYPE = os.getenv('DB_TYPE', 'postgres').lower()  # 'firebase' 또는 'postgres'

def initialize_database(db_type: str, force: bool = False):
    """데이터베이스 초기화"""
    try:
        if db_type == 'firebase':
            logger.info('Firebase 데이터베이스 초기화 시도...')

            # 환경 변수 확인
            if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                logger.warning('GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.')
                logger.warning('Firebase 인증 정보 파일 경로를 지정하세요.')
                return False

            # 초기화 스크립트 실행
            import subprocess
            result = subprocess.run(['python', 'initialize_firebase.py'], capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f'Firebase 초기화 실패: {result.stderr}')
                return False

            logger.info('Firebase 초기화 완료')
            return True

        elif db_type == 'postgres':
            logger.info('PostgreSQL 데이터베이스 초기화 시도...')

            # 환경 변수 확인
            if not all([
                os.getenv('POSTGRES_USER'),
                os.getenv('POSTGRES_PASSWORD'),
                os.getenv('POSTGRES_HOST')
            ]):
                logger.warning('PostgreSQL 연결 정보가 완전히 설정되지 않았습니다.')
                logger.warning('POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST 환경 변수를 설정하세요.')

            # 데이터베이스 초기화 스크립트 실행 (비동기)
            try:
                import asyncio

                from initialize_postgres import main as init_postgres

                # 임베딩 모델 사전 다운로드
                embedding_models = ['sentence-transformers/all-MiniLM-L6-v2']
                embedding_model_path = os.getenv('EMBEDDING_MODEL')
                if embedding_model_path:
                    embedding_models.insert(0, embedding_model_path)

                logger.info('임베딩 모델 다운로드 시작...')
                for model_name in embedding_models:
                    try:
                        # 임베딩 모델 미리 다운로드
                        from transformers import AutoModel
                        from transformers import AutoTokenizer
                        logger.info(f'임베딩 모델 다운로드 중: {model_name}')

                        # 캐시 디렉토리 설정
                        cache_dir = Path('model_cache') / 'embeddings'
                        cache_dir.mkdir(parents=True, exist_ok=True)

                        # 모델 및 토크나이저 다운로드
                        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

                        logger.info(f'임베딩 모델 다운로드 완료: {model_name}')
                        break
                    except Exception as e:
                        logger.warning(f'임베딩 모델 {model_name} 다운로드 실패: {e}')
                        continue

                # 데이터베이스 테이블 생성 및 샘플 데이터 추가
                if force:
                    logger.info('강제 초기화 모드: 기존 데이터를 덮어씁니다.')

                asyncio.run(init_postgres())
                logger.info('PostgreSQL 초기화 완료')
                return True
            except Exception as e:
                logger.error(f'PostgreSQL 초기화 실패: {e}', exc_info=True)
                return False
        else:
            logger.error(f'지원되지 않는 데이터베이스 타입: {db_type}')
            return False

    except Exception as e:
        logger.error(f'데이터베이스 초기화 중 오류 발생: {e}')
        return False

def check_environment():
    """환경 설정 확인"""
    # LLM 설정 확인
    llm_provider = os.getenv('LLM_PROVIDER', 'gemini').lower()
    llm_api_key = os.getenv('LLM_API_KEY', '')

    if llm_provider in ['gemini', 'hybrid'] and not llm_api_key:
        logger.warning('Gemini API 키가 설정되지 않았습니다. LLM_API_KEY 환경 변수를 설정하세요.')

    # MedLLaMA 설정 확인
    if llm_provider in ['medllama', 'hybrid']:
        medllama_path = os.getenv('MEDLLAMA_MODEL_PATH', '')
        if not medllama_path:
            logger.warning('의료 LLM 모델 경로가 설정되지 않았습니다. MEDLLAMA_MODEL_PATH 환경 변수를 설정하세요.')
            logger.info('공개 의료 LLM 모델을 자동으로 사용할 예정입니다.')

    # 데이터베이스 설정 확인
    db_type = DB_TYPE
    logger.info(f'데이터베이스 타입: {db_type}')

    # PubMed 설정 확인
    pubmed_email = os.getenv('PUBMED_EMAIL', '')
    if not pubmed_email:
        logger.warning('PubMed API 이메일이 설정되지 않았습니다. 제한된 기능으로 실행됩니다.')

    # Kaggle 설정 확인
    kaggle_username = os.getenv('KAGGLE_USERNAME', '')
    kaggle_key = os.getenv('KAGGLE_KEY', '')
    if not (kaggle_username and kaggle_key):
        logger.warning('Kaggle API 인증 정보가 설정되지 않았습니다. Kaggle 관련 기능이 제한됩니다.')

def create_env_template():
    """환경 변수 템플릿 파일 생성"""
    env_template_path = '.env.template'

    if os.path.exists(env_template_path):
        logger.info(f'환경 변수 템플릿 파일이 이미 존재합니다: {env_template_path}')
        return

    template = """# 기본 설정
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# LLM 제공자 설정 ('gemini', 'medllama', 'hybrid')
LLM_PROVIDER=hybrid
LLM_PRIMARY=gemini
LLM_API_KEY=your_gemini_api_key
GEMINI_MODEL_NAME=gemini-pro

# 데이터베이스 설정
DB_TYPE=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=medicaldb

# 의료 LLM 설정
MEDLLAMA_MODEL_NAME=meditron-7b
MEDLLAMA_MODEL_PATH=epfl-llm/meditron-7b
MEDLLAMA_DEVICE=cpu
MEDLLAMA_USE_QUANTIZATION=True
HUGGINGFACE_TOKEN=your_huggingface_token

# 임베딩 모델 설정
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# API 클라이언트 설정
PUBMED_EMAIL=your_email@example.com
PUBMED_API_KEY=your_pubmed_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# 백엔드 API 설정 (필요시)
AI_ANALYZER_API_ENDPOINT=http://localhost:8000
AI_ANALYZER_API_KEY=your_backend_api_key
"""

    with open(env_template_path, 'w') as f:
        f.write(template)

    logger.info(f'환경 변수 템플릿 파일이 생성되었습니다: {env_template_path}')

def preload_models():
    """LLM 및 임베딩 모델 사전 로드"""
    logger.info('모델 사전 로드 시작...')

    # 공통 캐시 디렉토리 설정
    cache_dir = Path('model_cache')
    cache_dir.mkdir(exist_ok=True)

    # 임베딩 모델 로드 (우선순위별)
    embedding_model_path = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    embedding_models = [
        embedding_model_path,
        'sentence-transformers/all-MiniLM-L6-v2',
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    ]

    for model_name in embedding_models:
        try:
            from transformers import AutoModel
            from transformers import AutoTokenizer
            logger.info(f'임베딩 모델 로드 중: {model_name}')

            # 모델 및 토크나이저 다운로드
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir/'embeddings')
            model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir/'embeddings')

            logger.info(f'임베딩 모델 로드 완료: {model_name}')
            break
        except Exception as e:
            logger.warning(f'임베딩 모델 {model_name} 로드 실패: {e}')
            continue

    # MedLLaMA/의료 LLM 모델 로드 (하이브리드 모드에서만)
    llm_provider = os.getenv('LLM_PROVIDER', '').lower()
    if llm_provider in ['medllama', 'hybrid']:
        try:
            medllama_path = os.getenv('MEDLLAMA_MODEL_PATH', 'epfl-llm/meditron-7b')

            if medllama_path:
                from transformers import AutoTokenizer
                logger.info(f'의료 LLM 모델 로드 중: {medllama_path}')

                # 모델 캐시 디렉토리
                med_cache_dir = cache_dir / 'medical_llm'
                med_cache_dir.mkdir(exist_ok=True)

                # 토크나이저만 미리 다운로드 (메모리 절약)
                tokenizer = AutoTokenizer.from_pretrained(
                    medllama_path,
                    cache_dir=med_cache_dir,
                    trust_remote_code=True
                )
                logger.info(f'의료 LLM 토크나이저 로드 완료: {medllama_path}')
        except Exception as e:
            logger.warning(f'의료 LLM 모델 미리 로드 실패: {e}')

def signal_handler(signum, frame):
    """시그널 핸들러"""
    logger.info(f'시그널 {signum} 수신. 서버를 종료합니다...')
    sys.exit(0)

def run_metrics_exporter(port=8001):
    """
    Prometheus 메트릭 익스포터 실행
    별도의 프로세스로 메트릭 서버를 구동합니다.
    """
    try:
        from prometheus_client import start_http_server

        # 메트릭 서버 시작
        start_http_server(port)
        logger.info(f'메트릭 서버가 포트 {port}에서 시작되었습니다.')

        return True
    except ImportError:
        logger.warning('prometheus_client 패키지가 설치되지 않았습니다. 메트릭 익스포터를 시작할 수 없습니다.')
        return False
    except Exception as e:
        logger.warning(f'메트릭 익스포터 시작 실패: {e}')
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='의료 AI 에이전트 서버 시작')
    parser.add_argument('--host', default=DEFAULT_HOST, help=f'서버 호스트 (기본값: {DEFAULT_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'서버 포트 (기본값: {DEFAULT_PORT})')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS, help=f'작업자 수 (기본값: {DEFAULT_WORKERS})')
    parser.add_argument('--reload', action='store_true', help='자동 리로드 활성화 (개발용)')
    parser.add_argument('--db-type', choices=['firebase', 'postgres'], default=DB_TYPE, help=f'데이터베이스 타입 (기본값: {DB_TYPE})')
    parser.add_argument('--init-db', action='store_true', help='서버 시작 전 데이터베이스 초기화')
    parser.add_argument('--force-init', action='store_true', help='데이터베이스 강제 초기화 (기존 데이터 덮어쓰기)')
    parser.add_argument('--log-level', default=DEFAULT_LOG_LEVEL, choices=['debug', 'info', 'warning', 'error', 'critical'], help=f'로깅 레벨 (기본값: {DEFAULT_LOG_LEVEL})')
    parser.add_argument('--generate-env', action='store_true', help='환경 변수 템플릿 파일 생성')
    parser.add_argument('--preload-models', action='store_true', help='모델 사전 로드')
    parser.add_argument('--metrics', action='store_true', help='메트릭 서버 활성화')
    parser.add_argument('--metrics-port', type=int, default=8001, help='메트릭 서버 포트 (기본값: 8001)')

    args = parser.parse_args()

    # 로깅 설정
    global logger
    logger = setup_logging(args.log_level)

    # 환경 변수 템플릿 생성
    if args.generate_env:
        create_env_template()
        if not args.init_db and not args.preload_models:
            return

    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 환경 변수 확인
    check_environment()

    # 메트릭 서버 시작 (요청 시)
    if args.metrics:
        run_metrics_exporter(args.metrics_port)

    # 모델 사전 로드 (요청 시)
    if args.preload_models:
        preload_models()

    # 데이터베이스 초기화 (요청 시)
    if args.init_db:
        logger.info(f'데이터베이스 초기화 시작 (타입: {args.db_type})...')
        initialize_database(args.db_type, args.force_init)

    # FastAPI 서버 시작
    logger.info(f'서버 시작: http://{args.host}:{args.port} (작업자 수: {args.workers})')

    # 서버 시작 정보 출력
    info = {
        'start_time': datetime.now().isoformat(),
        'host': args.host,
        'port': args.port,
        'workers': args.workers,
        'reload': args.reload,
        'db_type': args.db_type,
        'log_level': args.log_level,
    }
    logger.info(f'서버 설정: {json.dumps(info)}')

    # uvicorn 서버 시작
    uvicorn.run(
        'fastapi_app.main:app',
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers if not args.reload else 1,  # 리로드 모드에서는 단일 작업자 사용
    )

if __name__ == '__main__':
    main()
