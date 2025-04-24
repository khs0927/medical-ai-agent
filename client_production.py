#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
의료 AI 에이전트 클라이언트 (프로덕션 모드)
실제 API를 사용하는 서버에 연결하여 의료 질문에 대한 응답을 받습니다.
"""
import requests
import argparse
import sys
import json
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv(".env.production")

def consult(question, server_url="http://localhost:8000"):
    """의료 AI 에이전트에 질문을 전송하고 응답을 받습니다."""
    url = f"{server_url}/v1/consult"
    
    print(f"📨 질문을 전송 중입니다...")
    
    try:
        response = requests.post(
            url,
            json={"question": question},
            timeout=90  # 실제 API는 더 오래 걸릴 수 있음
        )
        
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"오류 ({response.status_code}): {response.text}"
    except Exception as e:
        return f"API 호출 오류: {e}"

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="의료 AI 에이전트 클라이언트 (프로덕션 모드)")
    parser.add_argument("--server", "-s", default="http://localhost:8000", help="서버 URL")
    parser.add_argument("--question", "-q", help="질문")
    parser.add_argument("--interactive", "-i", action="store_true", help="대화형 모드")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("의료 AI 에이전트 클라이언트 (프로덕션 모드)")
    print("=" * 80)
    print("🔴 주의: 이 모드에서는 실제 API가 호출되며 비용이 발생할 수 있습니다.")
    print("=" * 80)
    
    if args.interactive:
        print("\n대화형 모드 시작 (종료: Ctrl+C 또는 'exit' 입력)\n")
        
        while True:
            try:
                question = input("\n📝 질문: ")
                if question.lower() in ['exit', 'quit', 'q']:
                    break
                
                if not question.strip():
                    continue
                
                print("\n🤖 응답:")
                answer = consult(question, args.server)
                print(answer)
                print("\n" + "-" * 80)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"오류: {e}")
                break
        
        print("\n대화를 종료합니다.")
    
    elif args.question:
        answer = consult(args.question, args.server)
        print(f"\n🤖 응답:\n{answer}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 