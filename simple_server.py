#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
간단한 의료 AI 테스트 서버
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict, List

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="의료 AI 테스트 서버",
    description="테스트용 간단한 서버",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 및 템플릿 설정
try:
    app.mount("/static", StaticFiles(directory="ui/static"), name="static")
    templates = Jinja2Templates(directory="ui/templates")
    templates_available = True
except Exception as e:
    logger.warning(f"템플릿 디렉토리를 로드할 수 없습니다: {e}")
    templates_available = False

# API 모델
class QueryRequest(BaseModel):
    """쿼리 요청 모델"""
    query: str
    model: Optional[str] = "test-model"
    conversation_id: Optional[str] = None

# 루트 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def root():
    """루트 페이지"""
    if templates_available:
        return templates.TemplateResponse("index.html", {"request": {}})
    else:
        return HTMLResponse("<html><body><h1>의료 AI 테스트 서버</h1><p>API는 /api/query에서 사용할 수 있습니다.</p></body></html>")

# 헬스 체크 엔드포인트
@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "test_mode": True
    }

# 쿼리 엔드포인트
@app.post("/api/query")
async def process_query(request: QueryRequest):
    """쿼리 처리"""
    logger.info(f"쿼리 받음: {request.query}")
    
    # 테스트 응답 생성
    return {
        "response": f"테스트 응답: '{request.query}'에 대한 의학 정보입니다.",
        "sources": [
            {"title": "테스트 소스 1", "url": "https://example.com/1", "snippet": "테스트 데이터"},
            {"title": "테스트 소스 2", "url": "https://example.com/2", "snippet": "샘플 정보"}
        ],
        "conversation_id": request.conversation_id or "test-conversation-123",
        "model": request.model
    }

# 메인 함수
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"서버를 시작합니다: http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port) 