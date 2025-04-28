from __future__ import annotations

"""FastAPI 서버 (Phase 2)

`MedicalEnsemble` 래퍼를 활용하여 `/api/query` 단일 엔드포인트를 제공한다.
Phase 2: 데이터 계층(RAG) 추가 및 데이터 수집/문서 추가 엔드포인트 확장
"""

import time
import logging
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from models.ensemble import MedicalEnsemble
from rag.mock_retriever import MockRetriever, Document
from prompts.medical_prompts import MedicalPrompt
from data.pipeline import MedicalDataPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MediGenius API", version="0.2.0")

# 컴포넌트 초기화
retriever = MockRetriever()
ensemble = MedicalEnsemble()
data_pipeline = MedicalDataPipeline()
prompt_builder = MedicalPrompt(retriever)


# 요청/응답 모델 정의
class MedicalQuery(BaseModel):
    query: str
    patient_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    include_sources: bool = True


class MedicalResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    confidence: float
    processing_time: float


class PatientRecordInput(BaseModel):
    patient_id: str
    demographics: Dict[str, Any]
    medical_history: Optional[List[Dict[str, Any]]] = None
    medications: Optional[List[Dict[str, Any]]] = None
    lab_results: Optional[List[Dict[str, Any]]] = None
    vitals: Optional[Dict[str, Any]] = None


class DocumentInput(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class PatientRecordResponse(BaseModel):
    patient_id: str
    success: bool
    message: str


class DocumentResponse(BaseModel):
    doc_id: str
    success: bool
    message: str


# API 엔드포인트
@app.post("/api/query", response_model=MedicalResponse)
async def process_medical_query(query_data: MedicalQuery):
    """의료 질의 처리 엔드포인트 (Phase 2 - RAG 확장)"""
    if not query_data.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    start = time.time()
    
    try:
        # 1. 환자 데이터 검색 (있는 경우)
        patient_data = None
        if query_data.patient_id:
            patient_data = await retriever.get_patient_data(query_data.patient_id)
            if not patient_data:
                logger.warning(f"환자 ID '{query_data.patient_id}'에 대한 데이터를 찾을 수 없습니다.")
        
        # 2. 관련 의료 문헌 검색
        relevant_docs = await retriever.retrieve_documents(query_data.query, limit=3)
        
        # 3. 프롬프트 생성
        prompt = prompt_builder.create_dynamic_prompt(
            query_data.query,
            patient_data=patient_data
        )
        
        # 4. 컨텍스트 준비 (현재 구현에서는 사용되지 않지만 추후 확장 가능)
        context = {
            "patient_data": patient_data,
            "relevant_docs": [doc.dict() for doc in relevant_docs],
            "additional_context": query_data.context
        }
        
        # 5. 모델 예측 실행
        result = ensemble.predict(prompt, context=context)
        
        # 소요 시간 계산
        elapsed = time.time() - start
        
        # 응답 구성
        sources = []
        if query_data.include_sources and relevant_docs:
            for doc in relevant_docs:
                sources.append({
                    "id": doc.id,
                    "title": doc.title,
                    "content_snippet": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": doc.metadata,
                    "relevance_score": doc.score
                })
        
        return MedicalResponse(
            answer=result["response"],
            sources=sources,
            confidence=result["confidence"],
            processing_time=elapsed,
        )
    
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/patient/add", response_model=PatientRecordResponse)
async def add_patient_record(patient_data: PatientRecordInput):
    """환자 데이터 추가 (Phase 2)"""
    try:
        # 데이터 파이프라인 처리
        ehr_data = patient_data.dict()
        result = await data_pipeline.process_ehr_data(ehr_data)
        
        # retriever에 데이터 추가
        patient_id = result["doc_id"]
        await retriever.add_patient_data(patient_id, ehr_data)
        
        return PatientRecordResponse(
            patient_id=patient_id,
            success=True,
            message="Patient data added successfully"
        )
    
    except Exception as e:
        logger.error(f"Error adding patient data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error adding patient data: {str(e)}")


@app.post("/api/document/add", response_model=DocumentResponse)
async def add_document(document: DocumentInput):
    """의료 문서 추가 (Phase 2)"""
    try:
        # 데이터 파이프라인 처리
        metadata = document.metadata or {}
        if document.title:
            metadata["title"] = document.title
            
        result = await data_pipeline.process_medical_text(document.content, metadata)
        
        # retriever에 문서 추가
        doc_id = result["doc_id"]
        doc = Document(
            id=doc_id,
            title=document.title,
            content=document.content,
            metadata=document.metadata or {}
        )
        await retriever.add_document(doc)
        
        return DocumentResponse(
            doc_id=doc_id,
            success=True,
            message="Document added successfully"
        )
    
    except Exception as e:
        logger.error(f"Error adding document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")


@app.get("/api/documents", response_model=List[Dict[str, Any]])
async def list_documents():
    """저장된 문서 목록 조회 (Phase 2)"""
    documents = await data_pipeline.get_all_documents()
    return documents


@app.get("/api/patients", response_model=Dict[str, Dict[str, Any]])
async def list_patients():
    """저장된 환자 목록 조회 (Phase 2)"""
    patients = await data_pipeline.get_all_patients()
    return patients


@app.get("/health")
async def health_check():
    """상태 확인 엔드포인트"""
    return {"status": "ok"} 