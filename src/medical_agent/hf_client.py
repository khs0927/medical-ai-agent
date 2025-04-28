# -*- coding: utf-8 -*-
import logging
from typing import List, Dict, Any
import requests
import os

logger = logging.getLogger(__name__)

# Hugging Face API 키 설정
HF_API_KEY = os.getenv("HF_API_KEY", "your-api-key-here")
QWEN_MODEL = "Qwen/Qwen-7B-Chat"
GEMINI_MODEL = "google/gemini-pro"

def qwen_chat(messages: List[Dict[str, str]]) -> str:
    """Qwen 모델과 대화하는 함수"""
    logger.info("Qwen 모델 호출")
    
    # API 요청 헤더
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # API 요청 데이터
    data = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        # API 요청
        response = requests.post(
            "https://api-inference.huggingface.co/models/Qwen/Qwen-7B-Chat",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        # 응답 처리
        result = response.json()
        return result["generated_text"]
    except Exception as e:
        logger.error(f"Qwen 모델 호출 중 오류 발생: {str(e)}")
        return "Qwen 모델 응답 오류"

def gemini_chat(messages: List[Dict[str, str]]) -> str:
    """Gemini 모델과 대화하는 함수"""
    logger.info("Gemini 모델 호출")
    
    # API 요청 헤더
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # API 요청 데이터
    data = {
        "model": GEMINI_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        # API 요청
        response = requests.post(
            "https://api-inference.huggingface.co/models/google/gemini-pro",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        # 응답 처리
        result = response.json()
        return result["generated_text"]
    except Exception as e:
        logger.error(f"Gemini 모델 호출 중 오류 발생: {str(e)}")
        return "Gemini 모델 응답 오류"
