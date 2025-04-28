from __future__ import annotations

"""
가벼운 mock retriever 구현 (Phase 2)

실제 벡터 DB 연결 전에 미리 정의된 문서로 RAG 워크플로우를 테스트합니다.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Document(BaseModel):
    """검색 가능한 문서 표현"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = {}
    score: float = 0.0


class MockRetriever:
    """메모리 기반 Mock Retriever"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """기본 문서 세트로 초기화"""
        self.documents = self._load_default_documents()
        self.patient_data = self._load_mock_patient_data()

    def _load_default_documents(self) -> List[Document]:
        """기본 의학 문서 로드"""
        return [
            Document(
                id="doc1",
                title="고혈압 치료 가이드라인 2025",
                content="고혈압은 혈압이 지속적으로 140/90 mmHg 이상인 경우로 정의됩니다. "
                        "1단계 고혈압(140-159/90-99 mmHg)의 경우 생활습관 개선과 함께 약물 치료를 시작할 수 있으며, "
                        "2단계 고혈압(≥160/100 mmHg)은 즉시 약물 치료를 시작해야 합니다. "
                        "일차 약제로는 ACE 억제제, ARB, 칼슘 채널 차단제, 티아지드계 이뇨제가 권장됩니다.",
                metadata={
                    "type": "guideline",
                    "specialty": "cardiology",
                    "year": 2025,
                    "source": "대한고혈압학회"
                }
            ),
            Document(
                id="doc2",
                title="당뇨병 관리: 최신 연구 동향",
                content="제2형 당뇨병 환자의 경우 메트포르민이 일반적으로 1차 치료제로 사용됩니다. "
                        "혈당 조절이 충분하지 않을 경우 SGLT-2 억제제나 GLP-1 수용체 작용제를 추가할 수 있으며, "
                        "특히 심혈관질환 위험이 높은 환자에게 효과적입니다. "
                        "최근 연구에 따르면 일부 환자에서 저탄수화물 식이요법이 인슐린 감수성을 개선할 수 있습니다.",
                metadata={
                    "type": "research",
                    "specialty": "endocrinology",
                    "year": 2024,
                    "source": "Diabetes Care"
                }
            ),
            Document(
                id="doc3",
                title="두통의 감별 진단과 치료",
                content="편두통은 맥동성 통증, 오심, 광선공포증이 특징입니다. 트립탄 계열 약물이 급성기 치료에 효과적이며, "
                        "만성 편두통의 경우 보툴리눔 독소 주사나 CGRP 억제제가 예방 치료로 사용됩니다. "
                        "긴장성 두통은 압박감이 특징이며 일반 진통제가 효과적입니다. "
                        "갑작스러운 심한 두통('벼락두통')은 지주막하 출혈의 가능성이 있어 즉시 응급실을 방문해야 합니다.",
                metadata={
                    "type": "clinical",
                    "specialty": "neurology",
                    "year": 2023,
                    "source": "Neurology Today"
                }
            ),
            Document(
                id="doc4",
                title="COVID-19 후유증 관리",
                content="COVID-19 후유증(롱 코비드)은 급성기 이후 4주 이상 지속되는 증상으로, 피로, 호흡곤란, 인지장애, "
                        "관절통 등이 나타날 수 있습니다. 다학제적 접근이 중요하며, 단계적 운동 요법, 호흡 재활, "
                        "인지행동치료 등이 도움이 될 수 있습니다. 현재까지 특정 치료제는 없으나 증상별 대증치료를 제공합니다.",
                metadata={
                    "type": "clinical",
                    "specialty": "pulmonology",
                    "year": 2024,
                    "source": "Journal of Post-Acute COVID-19 Syndrome"
                }
            ),
            Document(
                id="doc5",
                title="불면증 치료 최신 지침",
                content="불면증 일차 치료로는 인지행동치료가 권장됩니다. 약물치료는 단기간 사용을 원칙으로 하며, "
                        "비벤조디아제핀계 수면제(Z-drugs)나 멜라토닌 수용체 작용제를 고려할 수 있습니다. "
                        "장기적인 벤조디아제핀 사용은 의존성, 인지기능 저하 등의 위험이 있어 주의가 필요합니다. "
                        "수면 위생 교육은 모든 불면증 환자에게 기본적으로 제공되어야 합니다.",
                metadata={
                    "type": "guideline",
                    "specialty": "psychiatry",
                    "year": 2025,
                    "source": "Sleep Medicine Reviews"
                }
            )
        ]

    def _load_mock_patient_data(self) -> Dict[str, Dict[str, Any]]:
        """모의 환자 데이터 생성"""
        return {
            "patient001": {
                "personal_info": {
                    "age": 45,
                    "gender": "male",
                    "height": 175,
                    "weight": 78
                },
                "vitals": {
                    "heart_rate": 72,
                    "blood_pressure": {
                        "systolic": 135,
                        "diastolic": 85
                    },
                    "oxygen_level": 98,
                    "temperature": 36.5
                },
                "medical_history": {
                    "conditions": ["Hypertension", "Type 2 Diabetes"],
                    "allergies": ["Penicillin"],
                    "surgeries": ["Appendectomy (2010)"],
                    "family_history": ["Father: Heart Disease", "Mother: Hypertension"]
                },
                "medications": [
                    {
                        "name": "Lisinopril",
                        "dosage": "10mg",
                        "frequency": "Once daily"
                    },
                    {
                        "name": "Metformin",
                        "dosage": "500mg",
                        "frequency": "Twice daily"
                    }
                ],
                "recent_measurements": [
                    {
                        "date": "2025-04-20",
                        "glucose": 142,
                        "hba1c": 6.8,
                        "blood_pressure": {
                            "systolic": 138,
                            "diastolic": 87
                        }
                    }
                ]
            },
            "patient002": {
                "personal_info": {
                    "age": 62,
                    "gender": "female",
                    "height": 162,
                    "weight": 65
                },
                "vitals": {
                    "heart_rate": 78,
                    "blood_pressure": {
                        "systolic": 142,
                        "diastolic": 88
                    },
                    "oxygen_level": 97,
                    "temperature": 36.7
                },
                "medical_history": {
                    "conditions": ["Osteoarthritis", "Hyperlipidemia"],
                    "allergies": ["Sulfa drugs"],
                    "surgeries": ["Knee replacement (2022)"],
                    "family_history": ["Mother: Breast cancer", "Sister: Rheumatoid arthritis"]
                },
                "medications": [
                    {
                        "name": "Atorvastatin",
                        "dosage": "20mg",
                        "frequency": "Once daily"
                    },
                    {
                        "name": "Acetaminophen",
                        "dosage": "500mg",
                        "frequency": "As needed for pain"
                    }
                ],
                "recent_measurements": [
                    {
                        "date": "2025-05-05",
                        "cholesterol": {
                            "total": 195,
                            "ldl": 110,
                            "hdl": 52
                        }
                    }
                ]
            }
        }

    async def retrieve_documents(self, query: str, limit: int = 3) -> List[Document]:
        """쿼리와 관련된 문서 검색 (단순 키워드 매칭)"""
        logger.info(f"문서 검색: '{query}'")
        
        # 검색어 전처리
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        # 각 문서의 관련성 점수 계산 (단순 용어 빈도)
        scored_docs = []
        for doc in self.documents:
            content_lower = doc.content.lower()
            title_lower = doc.title.lower()
            
            # 단순 용어 매칭 점수 계산
            score = 0
            for term in query_terms:
                # 제목에서 일치하면 가중치 2
                if term in title_lower:
                    score += 2
                # 내용에서 일치하면 가중치 1
                count = content_lower.count(term)
                score += count
            
            if score > 0:
                # 점수가 있는 경우에만 추가
                doc_copy = doc.copy()
                doc_copy.score = score
                scored_docs.append(doc_copy)
        
        # 점수로 정렬하고 상위 문서 반환
        sorted_docs = sorted(scored_docs, key=lambda d: d.score, reverse=True)
        return sorted_docs[:limit]
    
    async def get_patient_data(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """환자 ID로 환자 데이터 검색"""
        logger.info(f"환자 데이터 검색: {patient_id}")
        return self.patient_data.get(patient_id)
    
    async def add_document(self, document: Document) -> str:
        """새 문서 추가 (메모리에만 저장)"""
        # ID가 없으면 생성
        if not document.id:
            document.id = f"doc{len(self.documents) + 1}"
        
        # 기존 문서 업데이트 또는 추가
        for i, doc in enumerate(self.documents):
            if doc.id == document.id:
                self.documents[i] = document
                return document.id
        
        # 새 문서 추가
        self.documents.append(document)
        return document.id
    
    async def add_patient_data(self, patient_id: str, data: Dict[str, Any]) -> str:
        """환자 데이터 추가 또는 업데이트"""
        self.patient_data[patient_id] = data
        return patient_id 