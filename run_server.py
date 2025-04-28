#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
의료 AI 에이전트 서버 실행 스크립트
테스트 모드로 서버를 시작합니다.
"""
import os
import sys
import uvicorn
from dotenv import load_dotenv

def main():
    # 환경 변수 로드
    load_dotenv(".env.test")
    
    # 테스트 모드 설정
    os.environ["TEST_MODE"] = "true"
    
    print("=" * 80)
    print("의료 AI 에이전트 서버 시작 (테스트 모드)")
    print("=" * 80)
    print("서버 URL: http://localhost:8000")
    print("API 엔드포인트: /v1/consult")
    print("Health 체크: /healthz")
    print("서버 종료: Ctrl+C")
    print("=" * 80)
    
    # 서버 실행
    uvicorn.run(
        "fastapi_app.main:app", 
        host="0.0.0.0", 
        port=8001,
        reload=True
    )

if __name__ == "__main__":
    main() 