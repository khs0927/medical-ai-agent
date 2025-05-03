#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 3.12와 호환되는 의료 AI 서버 실행 스크립트
"""

import os
import sys
import subprocess
import argparse

def run_server(test_mode=True):  # 기본값을 True로 변경 (항상 테스트 모드로 실행)
    """서버 실행 함수"""
    # 환경 변수 설정 - 항상 테스트 모드로 실행 (protobuf 호환성 문제 해결될 때까지)
    os.environ['TEST_MODE'] = 'true'
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    
    # 서버 스크립트 경로
    SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui', 'simple_server.py')
    
    print(f"의료 AI 서버 시작 중... (테스트 모드)")
    print("주소: http://localhost:8080/")
    
    try:
        # 서버 실행
        subprocess.run([sys.executable, SERVER_SCRIPT], env=os.environ)
    except KeyboardInterrupt:
        print("\n서버가 종료되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    # 명령행 인자 처리
    parser = argparse.ArgumentParser(description="의료 AI 서버 실행")
    parser.add_argument("--test", action="store_true", help="테스트 모드로 실행 (실제 AI 모델을 사용하지 않음)")
    args = parser.parse_args()
    
    # 서버 실행
    run_server(test_mode=args.test) 