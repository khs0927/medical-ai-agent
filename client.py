#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
의료 AI 에이전트 클라이언트
API를 호출하여 사용자의 질문에 대한 응답을 받습니다.
"""
import requests
import argparse
import sys
import json

def consult(question, server_url="http://localhost:8000"):
    """의료 AI 에이전트에 질문을 전송하고 응답을 받습니다."""
    url = f"{server_url}/v1/consult"
    
    try:
        response = requests.post(
            url,
            json={"question": question},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"오류 ({response.status_code}): {response.text}"
    except Exception as e:
        return f"API 호출 오류: {e}"

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="의료 AI 에이전트 클라이언트")
    parser.add_argument("--server", "-s", default="http://localhost:8000", help="서버 URL")
    parser.add_argument("--question", "-q", help="질문")
    parser.add_argument("--interactive", "-i", action="store_true", help="대화형 모드")
    
    args = parser.parse_args()
    
    if args.interactive:
        print("=" * 60)
        print("의료 AI 에이전트 대화형 모드 (종료: Ctrl+C 또는 'exit' 입력)")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n질문: ")
                if question.lower() in ['exit', 'quit', 'q']:
                    break
                
                if not question.strip():
                    continue
                
                print("\n응답:")
                answer = consult(question, args.server)
                print(answer)
                print("\n" + "-" * 60)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"오류: {e}")
                break
        
        print("\n대화를 종료합니다.")
    
    elif args.question:
        answer = consult(args.question, args.server)
        print(answer)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 