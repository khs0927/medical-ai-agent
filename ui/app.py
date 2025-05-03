#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
의료 AI 에이전트 테스트 웹 인터페이스

FastAPI 기반 웹 서버와 API 엔드포인트를 구현합니다.
이 앱은, 웹 UI와 의료 AI 에이전트 간의 통신을 중계합니다.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# 현재 디렉토리를 Python 경로에 추가하여 agent.py를 임포트할 수 있게 합니다
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 의료 에이전트 모듈 임포트
from agent import Agent
from db.firebase_db import FirebaseDB

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(title="Medical AI Agent UI")

# 정적 파일과 템플릿 설정
app.mount("/static", StaticFiles(directory="ui/static"), name="static")
templates = Jinja2Templates(directory="ui/templates")

# 에이전트 인스턴스 및 DB 연결 생성
try:
    # Firebase DB 연결
    firebase_db = FirebaseDB()
    # 에이전트 생성 (기본 모델 사용)
    default_agent = Agent(db=firebase_db)
    logger.info('에이전트 초기화 완료')
except Exception as e:
    logger.error('에이전트 초기화 실패: %s', e)
    default_agent = None

# 요청 모델 정의
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    use_search: bool = True
    model: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """메인 페이지를 렌더링합니다."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """채팅 API 엔드포인트"""
    if default_agent is None:
        raise HTTPException(status_code=500, detail="에이전트가 초기화되지 않았습니다.")
    
    try:
        # 사용자 메시지 로깅
        logger.info('사용자 메시지: %s', chat_request.message)
        
        # 대화 기록 형식 변환
        conversation_history = []
        for msg in chat_request.history:
            if msg['role'] == 'user':
                conversation_history.append({"role": "user", "content": msg['content']})
            elif msg['role'] == 'agent':
                conversation_history.append({"role": "assistant", "content": msg['content']})
        
        # 에이전트에게 질의
        response = default_agent.chat(
            message=chat_request.message,
            history=conversation_history,
            use_search=chat_request.use_search,
        )
        
        # 응답 로깅
        logger.info('에이전트 응답: %s', response['content'] if isinstance(response, dict) else response)
        
        # 응답 형식 변환
        if isinstance(response, dict):
            answer = response.get('content', '')
            sources = response.get('sources', [])
        else:
            answer = response
            sources = []
            
        return ChatResponse(answer=answer, sources=sources)
        
    except Exception as e:
        logger.error('에이전트 응답 중 오류 발생: %s', e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """상태 확인 엔드포인트"""
    status = "healthy" if default_agent is not None else "unhealthy"
    return {"status": status}

def main():
    """앱 실행 함수"""
    uvicorn.run("ui.app:app", host="0.0.0.0", port=8080, reload=True)

if __name__ == "__main__":
    main() 