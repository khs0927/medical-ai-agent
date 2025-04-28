from __future__ import annotations

"""
의료 데이터 처리 파이프라인 (Phase 2)

문서, 환자 데이터 등의 정보를 처리하고 DB에 저장하기 위한 파이프라인.
MVP에서는 메모리 기반 사전 처리만 지원하고 추후 확장.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PatientData(BaseModel):
    """환자 데이터 모델"""
    patient_id: str
    age: int
    gender: str
    medical_history: List[Dict[str, Any]] = []
    medications: List[Dict[str, Any]] = []
    lab_results: List[Dict[str, Any]] = []
    imaging_data: List[Dict[str, Any]] = []
    vitals: Dict[str, Any] = {}


class MedicalDataPipeline:
    """의료 데이터 파이프라인 (경량 MVP 구현)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """데이터 파이프라인 초기화"""
        self.config = config or {}
        self.retriever = None  # 실제 구현에서는 여기에 retriever 연결
        self.in_memory_data = {
            "documents": [],
            "patients": {}
        }

    async def process_ehr_data(self, ehr_data: Dict[str, Any]) -> Dict[str, Any]:
        """전자건강기록(EHR) 데이터 처리
        
        Args:
            ehr_data: 환자 EHR 데이터
            
        Returns:
            처리된 데이터와 메타데이터
        """
        try:
            # 1. 데이터 구조화 및 정규화
            structured_data = self._structure_ehr(ehr_data)
            
            # 2. 중요 의료 엔티티 추출
            entities = self._extract_medical_entities(structured_data)
            
            # 3. 메모리에 저장 (실제 구현에서는 DB에 저장)
            doc_id = structured_data.get("patient_id", f"patient_{len(self.in_memory_data['patients']) + 1}")
            self.in_memory_data["patients"][doc_id] = structured_data
            
            return {
                "doc_id": doc_id,
                "structured_data": structured_data,
                "entities": entities
            }
        
        except Exception as e:
            logger.error(f"EHR 데이터 처리 중 오류: {e}", exc_info=True)
            raise

    def _structure_ehr(self, ehr_data: Dict[str, Any]) -> Dict[str, Any]:
        """EHR 데이터 구조화 및 전처리"""
        # MVP에서는 최소한의 처리만 수행
        structured = {}
        
        # 환자 기본 정보
        structured["patient_id"] = ehr_data.get("patient_id", "")
        
        # 인구통계학적 정보
        demographics = ehr_data.get("demographics", {})
        structured["demographics"] = {
            "age": demographics.get("age", 0),
            "gender": demographics.get("gender", "unknown"),
            "height": demographics.get("height", 0),
            "weight": demographics.get("weight", 0),
            "bmi": demographics.get("bmi", 0)
        }
        
        # 의료 이력
        structured["medical_history"] = ehr_data.get("medical_history", [])
        
        # 약물 처방
        structured["medications"] = ehr_data.get("medications", [])
        
        # 검사 결과
        structured["lab_results"] = ehr_data.get("lab_results", [])
        
        # 활력 징후
        structured["vitals"] = ehr_data.get("vitals", {})
        
        # 처리 메타데이터
        structured["processed_at"] = datetime.now().isoformat()
        
        return structured

    def _extract_medical_entities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """구조화된 데이터에서 중요 의료 개체 추출
        
        예) 질병명, 약물명, 증상 등을 식별하여 나중에 검색하기 쉽게 함
        """
        # MVP에서는 단순 추출만 수행 (조건명, 약물명 등)
        entities = []
        
        # 질병 추출
        if "medical_history" in data:
            for condition in data.get("medical_history", []):
                if isinstance(condition, dict) and "condition" in condition:
                    entities.append({
                        "type": "condition",
                        "value": condition["condition"],
                        "source": "medical_history"
                    })
                elif isinstance(condition, str):
                    entities.append({
                        "type": "condition",
                        "value": condition,
                        "source": "medical_history"
                    })
        
        # 약물 추출
        for med in data.get("medications", []):
            if isinstance(med, dict) and "name" in med:
                entities.append({
                    "type": "medication",
                    "value": med["name"],
                    "dosage": med.get("dosage", ""),
                    "source": "medications"
                })
            elif isinstance(med, str):
                entities.append({
                    "type": "medication",
                    "value": med,
                    "source": "medications"
                })
        
        # 알레르기 추출
        for allergy in data.get("allergies", []):
            if isinstance(allergy, dict) and "name" in allergy:
                entities.append({
                    "type": "allergy",
                    "value": allergy["name"],
                    "source": "allergies"
                })
            elif isinstance(allergy, str):
                entities.append({
                    "type": "allergy",
                    "value": allergy,
                    "source": "allergies"
                })
        
        return entities

    async def process_medical_text(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """의학 텍스트(논문, 가이드라인 등) 처리
        
        Args:
            text: 원문 텍스트
            metadata: 텍스트 메타데이터 (제목, 저자 등)
            
        Returns:
            처리된 문서 정보
        """
        try:
            # 1. 텍스트 전처리
            cleaned_text = self._preprocess_text(text)
            
            # 2. 중요 의학 용어 추출 (MVP에서는 생략)
            medical_terms = []
            
            # 3. 메모리에 저장 (실제 구현에서는 DB에 저장)
            doc_id = metadata.get("id", f"doc_{len(self.in_memory_data['documents']) + 1}")
            document = {
                "id": doc_id,
                "title": metadata.get("title", "Untitled"),
                "content": cleaned_text,
                "metadata": metadata,
                "processed_at": datetime.now().isoformat()
            }
            self.in_memory_data["documents"].append(document)
            
            return {
                "doc_id": doc_id,
                "document": document,
                "medical_terms": medical_terms
            }
        
        except Exception as e:
            logger.error(f"의학 텍스트 처리 중 오류: {e}", exc_info=True)
            raise

    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        # MVP에서는 기본적인 정리만 수행
        cleaned = text.strip()
        return cleaned

    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """저장된 모든 문서 조회"""
        return self.in_memory_data["documents"]

    async def get_all_patients(self) -> Dict[str, Dict[str, Any]]:
        """저장된 모든 환자 정보 조회"""
        return self.in_memory_data["patients"] 