from __future__ import annotations

"""
의학 프롬프트 생성기 (Phase 2)

의학 질의에 맞는 효과적인 프롬프트 생성
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class MedicalPrompt:
    """의학 프롬프트 생성 클래스"""

    def __init__(self, retriever=None):
        """초기화"""
        self.retriever = retriever
        logger.info("MedicalPrompt 초기화 완료")

    def create_dynamic_prompt(self, query: str, patient_data: Optional[Dict[str, Any]] = None) -> str:
        """동적 프롬프트 생성 (RAG 기능 포함)
        
        Args:
            query: 사용자 쿼리
            patient_data: 환자 데이터 (있는 경우)
            
        Returns:
            생성된 프롬프트
        """
        # 기본 시스템 프롬프트
        system_prompt = self._get_system_prompt()
        
        # 환자 정보 포맷
        patient_context = self._format_patient_data(patient_data) if patient_data else ""
        
        # 현재 시간 (필요시)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 최종 프롬프트 구성
        full_prompt = f"""{system_prompt}

현재 시간: {current_time}

{patient_context}

사용자 질문: {query}

환자 관련 정보와 최신 의학 지식에 기반해 답변해주세요. 정보가 불충분하거나 불확실한 경우 솔직하게 말해주세요.
"""
        
        return full_prompt

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 템플릿 반환"""
        return """당신은 MediGenius라는 의료 AI 어시스턴트입니다. 항상 최신 의학 지식과 증거 기반 의학을 바탕으로 정확하고 유용한 정보를 제공합니다.

다음 원칙을 따라 응답하세요:
1. 정확성: 검증된 의학 지식만 제공하고 불확실한 내용은 명시적으로 표현하세요.
2. 공감: 환자의 상황을 공감하고 존중하는 어조로 소통하세요.
3. 명확성: 의학 용어를 사용할 때는 가능한 쉽게 설명하세요.
4. 책임감: 당신은 진단을 내리거나 특정 치료를 처방할 수 없습니다. 필요시 의료 전문가와 상담을 권유하세요.
5. 개인정보 보호: 환자 데이터를 엄격히 보호하세요.

응답 형식:
- 질문에 대한 직접적인 답변
- 관련 의학 정보 및 설명
- 필요한 경우 추가 조치 권장사항
- 정보의 한계점 명시

중요: 당신의 답변은 참고용이며 의사의 진료를 대체할 수 없음을 항상 강조하세요."""

    def _format_patient_data(self, patient_data: Dict[str, Any]) -> str:
        """환자 데이터 포맷팅"""
        if not patient_data:
            return ""
        
        formatted_data = ["환자 정보:"]
        
        # 기본 정보
        if "personal_info" in patient_data:
            pi = patient_data["personal_info"]
            if "age" in pi:
                formatted_data.append(f"- 나이: {pi.get('age')}세")
            if "gender" in pi:
                gender = "남성" if pi.get('gender') == "male" else "여성"
                formatted_data.append(f"- 성별: {gender}")
            if "height" in pi and "weight" in pi:
                formatted_data.append(f"- 신체: 키 {pi.get('height')}cm, 체중 {pi.get('weight')}kg")
        
        # 활력징후
        if "vitals" in patient_data:
            vitals = patient_data["vitals"]
            vitals_info = []
            
            if "heart_rate" in vitals:
                vitals_info.append(f"심박수 {vitals.get('heart_rate')}bpm")
            
            if "blood_pressure" in vitals and isinstance(vitals["blood_pressure"], dict):
                bp = vitals["blood_pressure"]
                if "systolic" in bp and "diastolic" in bp:
                    vitals_info.append(f"혈압 {bp.get('systolic')}/{bp.get('diastolic')}mmHg")
            
            if "oxygen_level" in vitals:
                vitals_info.append(f"산소포화도 {vitals.get('oxygen_level')}%")
            
            if "temperature" in vitals:
                vitals_info.append(f"체온 {vitals.get('temperature')}°C")
            
            if vitals_info:
                formatted_data.append(f"- 활력징후: {', '.join(vitals_info)}")
        
        # 병력
        if "medical_history" in patient_data:
            med_history = patient_data["medical_history"]
            if isinstance(med_history, dict) and "conditions" in med_history:
                conditions = med_history["conditions"]
                if conditions and isinstance(conditions, list):
                    formatted_data.append(f"- 병력: {', '.join(conditions)}")
            elif isinstance(med_history, list):
                conditions = []
                for item in med_history:
                    if isinstance(item, str):
                        conditions.append(item)
                    elif isinstance(item, dict) and "condition" in item:
                        conditions.append(item["condition"])
                if conditions:
                    formatted_data.append(f"- 병력: {', '.join(conditions)}")
        
        # 약물
        if "medications" in patient_data:
            meds = patient_data["medications"]
            if isinstance(meds, list) and meds:
                med_list = []
                for med in meds:
                    if isinstance(med, str):
                        med_list.append(med)
                    elif isinstance(med, dict) and "name" in med:
                        med_str = med["name"]
                        if "dosage" in med:
                            med_str += f" {med['dosage']}"
                        if "frequency" in med:
                            med_str += f" ({med['frequency']})"
                        med_list.append(med_str)
                if med_list:
                    formatted_data.append(f"- 현재 약물: {', '.join(med_list)}")
        
        # 알레르기
        if "allergies" in patient_data:
            allergies = patient_data["allergies"]
            if isinstance(allergies, list) and allergies:
                formatted_data.append(f"- 알레르기: {', '.join(allergies)}")
        
        # 최근 측정치 (간략히)
        if "recent_measurements" in patient_data:
            measurements = patient_data["recent_measurements"]
            if isinstance(measurements, list) and measurements:
                latest = measurements[0]  # 가장 최근 측정치
                meas_list = []
                
                if "date" in latest:
                    meas_list.append(f"측정일: {latest['date']}")
                
                if "glucose" in latest:
                    meas_list.append(f"혈당: {latest['glucose']}mg/dL")
                
                if "hba1c" in latest:
                    meas_list.append(f"당화혈색소: {latest['hba1c']}%")
                
                if "cholesterol" in latest and isinstance(latest["cholesterol"], dict):
                    chol = latest["cholesterol"]
                    if "total" in chol:
                        meas_list.append(f"총 콜레스테롤: {chol['total']}mg/dL")
                    if "ldl" in chol:
                        meas_list.append(f"LDL: {chol['ldl']}mg/dL")
                    if "hdl" in chol:
                        meas_list.append(f"HDL: {chol['hdl']}mg/dL")
                
                if meas_list:
                    formatted_data.append(f"- 최근 검사: {', '.join(meas_list)}")
        
        return "\n".join(formatted_data)
    
    def create_rag_prompt(self, query: str, relevant_docs: List[Dict[str, Any]], patient_data: Optional[Dict[str, Any]] = None) -> str:
        """RAG 프롬프트 생성 (검색된 문서 포함)
        
        Args:
            query: 사용자 쿼리
            relevant_docs: 검색된 관련 문서 목록
            patient_data: 환자 데이터 (있는 경우)
            
        Returns:
            RAG 컨텍스트가 포함된 프롬프트
        """
        # 기본 프롬프트 생성
        base_prompt = self.create_dynamic_prompt(query, patient_data)
        
        # 관련 문서 없는 경우
        if not relevant_docs:
            return f"{base_prompt}\n\n관련 의학 정보: 관련 정보가 검색되지 않았습니다."
        
        # 관련 문서 포맷팅
        doc_sections = ["관련 의학 정보:"]
        
        for i, doc in enumerate(relevant_docs, 1):
            doc_title = doc.get("title", f"문서 {i}")
            doc_content = doc.get("content", "")
            if len(doc_content) > 1000:
                doc_content = doc_content[:997] + "..."
                
            doc_sections.append(f"[문서 {i}] {doc_title}\n{doc_content}\n")
        
        rag_context = "\n".join(doc_sections)
        
        # 최종 RAG 프롬프트
        rag_prompt = f"{base_prompt}\n\n{rag_context}\n\n위 정보와 당신의 의학 지식을 바탕으로 질문에 답변하세요."
        return rag_prompt
    
    def create_synthesis_prompt(self, query: str, patient_data: Optional[Dict[str, Any]], model_outputs: Dict[str, Any]) -> str:
        """다양한 모델 출력을 합성하기 위한 프롬프트
        
        Args:
            query: 원래 사용자 쿼리
            patient_data: 환자 데이터 (있는 경우)
            model_outputs: 각 모델별 출력 결과
            
        Returns:
            합성 프롬프트
        """
        # 기본 프롬프트 생성
        base_prompt = self.create_dynamic_prompt(query, patient_data)
        
        # 각 모델 결과 포맷팅
        model_sections = ["각 의료 AI 모델 예측 결과:"]
        
        for model_name, output in model_outputs.items():
            if isinstance(output, str):
                model_sections.append(f"[{model_name}]\n{output}\n")
            elif isinstance(output, dict) and "response" in output:
                model_sections.append(f"[{model_name}]\n{output['response']}\n")
        
        model_context = "\n".join(model_sections)
        
        # 최종 합성 프롬프트
        synthesis_prompt = f"""{base_prompt}

{model_context}

위 다양한 모델의 응답을 종합하여 가장 정확하고 유용한 답변을 제공하세요. 모델 간 불일치가 있다면 그 이유를 설명하고 가장 신뢰할 수 있는 정보를 제시하세요. 어떤 모델이 특정 측면에서 더 정확한지 의견을 제시할 수 있습니다.

최종 답변:"""
        
        return synthesis_prompt 