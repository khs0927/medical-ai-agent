#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 3.12 호환 의료 AI 서버 간소화 실행 스크립트

uvicorn을 직접 사용하여 FastAPI 앱을 실행합니다.
Click 패키지 의존성을 완전히 제거한 버전입니다.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# 전역 변수
HOST = "0.0.0.0"
PORT = 8080
TEST_MODE = False  # 실제 AI 모드 활성화

def run_server():
    """간소화된 서버 실행 함수"""
    print(f"의료 AI 서버 시작 중... (Python 3.12 호환 모드)")
    print(f"주소: http://{HOST}:{PORT}/")
    
    # 서버 시작 안내
    print(f"서버가 시작되었습니다. http://{HOST}:{PORT}/ 에서 접속 가능합니다.")
    print(f"모드: {'테스트' if TEST_MODE else '실제 AI'}")
    print("종료하려면 Ctrl+C를 누르세요.")
    
    # 에이전트 초기화
    print("의료 AI 에이전트 로드 중...")
    
    # 테스트 모드 환경 변수 설정 (false = 실제 AI 모드)
    os.environ["TEST_MODE"] = "false"
    
    # 간소화된 HTTP 서버 실행
    try:
        from ui.simple_server import run
        
        # 간단한 HTTP 서버 시작
        print(f"실제 AI 모드로 서버를 실행합니다.")
        run(port=PORT)
        
    except KeyboardInterrupt:
        print("\n서버가 종료되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 명령행 인자 처리
    if len(sys.argv) > 1:
        if "--port" in sys.argv:
            try:
                port_idx = sys.argv.index("--port") + 1
                PORT = int(sys.argv[port_idx])
            except (ValueError, IndexError):
                print("--port 뒤에 정수값을 입력하세요.")
                sys.exit(1)
                
        if "--host" in sys.argv:
            try:
                host_idx = sys.argv.index("--host") + 1
                HOST = sys.argv[host_idx]
            except (ValueError, IndexError):
                print("--host 뒤에 호스트명을 입력하세요.")
                sys.exit(1)
                
        if "--test" in sys.argv:
            TEST_MODE = True
            os.environ["TEST_MODE"] = "true"
            
        if "--prod" in sys.argv:
            TEST_MODE = False
            os.environ["TEST_MODE"] = "false"
    
    # 서버 실행
    run_server()
