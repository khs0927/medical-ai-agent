#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
의료 AI 에이전트 서버 실행 스크립트 (프로덕션 모드)
실제 API를 호출하는 프로덕션 모드로 서버를 시작합니다.
"""
import os
import sys
import uvicorn
from dotenv import load_dotenv

def main():
    # 환경 변수 로드
    load_dotenv(".env.production")
    
    # 테스트 모드 명시적 비활성화
    os.environ["TEST_MODE"] = "false"
    
    # API 키 확인
    gemini_key = os.environ.get("GEMINI_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not gemini_key or gemini_key == "":
        print("⚠️ 경고: GEMINI_API_KEY가 설정되지 않았습니다.")
        print("Google AI Studio에서 API 키를 발급받아 .env.production 파일에 설정하세요.")
        print("참고: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    if not hf_token or hf_token == "":
        print("⚠️ 경고: 현재 HF_TOKEN이 설정되지 않았습니다.")
        print("일부 기능은 모의 데이터로 대체될 수 있습니다.")
        print("HuggingFace 토큰 발급: https://huggingface.co/settings/tokens")
    
    print("=" * 80)
    print("의료 AI 에이전트 서버 시작 (프로덕션 모드)")
    print("=" * 80)
    print("🔴 주의: 이 모드에서는 실제 API가 호출되며 비용이 발생할 수 있습니다.")
    print("서버 URL: http://localhost:8000")
    print("API 엔드포인트: /v1/consult")
    print("Health 체크: /healthz")
    print("서버 종료: Ctrl+C")
    print("=" * 80)
    
    # 서버 실행
    uvicorn.run(
        "src.medical_agent.__main__:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )

if __name__ == "__main__":
    main() 