#!/usr/bin/env python3
"""
PostgreSQL 데이터베이스 초기화 스크립트

필요한 테이블을 생성하고 샘플 데이터를 추가합니다.
"""
import asyncio
import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from db.postgres_config import Base
from db.postgres_db import PostgresDB

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# PostgreSQL 접속 정보
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'medicaldb')

# SQLAlchemy 연결 문자열
DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'


async def create_tables():
    """데이터베이스 테이블 생성"""
    logger.info('데이터베이스 테이블 생성 시작...')
    
    try:
        # 엔진 생성
        engine = create_engine(DATABASE_URL)
        
        # 모든 모델의 테이블 생성
        Base.metadata.create_all(engine)
        
        logger.info('데이터베이스 테이블 생성 완료')
    except SQLAlchemyError as e:
        logger.error(f'테이블 생성 중 오류 발생: {e}', exc_info=True)
        raise


async def add_sample_data():
    """샘플 데이터 추가"""
    logger.info('샘플 데이터 추가 시작...')
    
    try:
        # PostgresDB 인스턴스 생성
        db = PostgresDB()
        
        # 샘플 의학 문헌 추가
        await add_sample_medical_literature(db)
        
        # 샘플 임상 가이드라인 추가
        await add_sample_clinical_guidelines(db)
        
        # 샘플 환자 기록 추가
        await add_sample_patient_records(db)
        
        logger.info('샘플 데이터 추가 완료')
    except Exception as e:
        logger.error(f'샘플 데이터 추가 중 오류 발생: {e}', exc_info=True)
        raise


async def add_sample_medical_literature(db):
    """샘플 의학 문헌 추가"""
    logger.info('샘플 의학 문헌 추가...')
    
    # 샘플 문헌 1: 당뇨병 관련
    literature1 = {
        'title': '최신 제2형 당뇨병 치료 가이드라인: 체계적 리뷰',
        'authors': ['김철수', '이영희', '박지영'],
        'publication_date': '2023-06-15',
        'journal': '대한당뇨병학회지',
        'abstract': '이 연구는 제2형 당뇨병 치료에 관한 최신 가이드라인을 체계적으로 검토했습니다. 메트포르민은 대부분의 가이드라인에서 여전히 1차 치료제로 권장되지만, SGLT2 억제제와 GLP-1 수용체 작용제의 역할이 심혈관 위험이 있는 환자에게 더 중요해지고 있습니다. 개인화된 치료 접근법이 강조되며, 환자 중심의 관리가 주요 원칙으로 부상하고 있습니다.',
        'doi': '10.1234/kda.2023.123',
        'mesh_terms': ['당뇨병, 제2형', '약물요법', '메트포르민', 'SGLT2 억제제', 'GLP-1 수용체 작용제'],
        'metadata': {
            'citations': 24,
            'study_type': '체계적 리뷰',
            'keywords': ['당뇨병', '가이드라인', '약물치료', '심혈관 위험']
        }
    }
    
    # 샘플 문헌 2: 고혈압 관련
    literature2 = {
        'title': '고혈압 관리에서 생활 습관 중재의 효과',
        'authors': ['정민수', '최수영', '강성호'],
        'publication_date': '2023-04-22',
        'journal': '한국고혈압학회지',
        'abstract': '본 연구는 고혈압 환자에서 다양한 생활 습관 중재의 효과를 평가했습니다. 체중 감량, 다쉬(DASH) 식이 요법, 나트륨 섭취 감소, 정기적인 신체 활동 및 적절한 알코올 소비가 고혈압 관리에 효과적인 것으로 밝혀졌습니다. 특히, 여러 개입을 결합하면 약물 요법이 필요하지 않거나 항고혈압제 용량을 줄일 수 있는 상당한 혈압 감소 효과가 있는 것으로 나타났습니다.',
        'doi': '10.5678/ksh.2023.456',
        'mesh_terms': ['고혈압', '생활 습관', '다쉬 식이', '체중 감량', '신체 활동'],
        'metadata': {
            'citations': 15,
            'study_type': '메타분석',
            'keywords': ['고혈압', '생활 습관', '비약물적 중재', '혈압 감소']
        }
    }
    
    # 샘플 문헌 3: COVID-19 관련
    literature3 = {
        'title': 'COVID-19 후 장기 증상: 체계적 검토 및 메타분석',
        'authors': ['한지영', '조현우', '송미라'],
        'publication_date': '2023-07-10',
        'journal': '감염병학회지',
        'abstract': '이 연구는 COVID-19 감염 후 장기 증상(롱 코비드)의 유병률과 위험 요인을 평가했습니다. 분석 결과, 환자의 약 30%가 감염 후 3개월 이상 적어도 하나의 증상을 경험했으며, 피로, 호흡 곤란, 인지 장애가 가장 흔한 증상으로 나타났습니다. 초기 감염 중 중증도, 여성, 기존 건강 상태가 장기 증상 발생의 주요 위험 요인이었습니다.',
        'doi': '10.9012/kid.2023.789',
        'mesh_terms': ['COVID-19', '후유증', '만성 증상', '피로', '인지 장애'],
        'metadata': {
            'citations': 42,
            'study_type': '체계적 리뷰/메타분석',
            'keywords': ['롱 코비드', 'COVID-19', '후유증', '만성 증상']
        }
    }
    
    # 데이터베이스에 추가
    doc_id1 = await db.add_medical_literature(literature1)
    doc_id2 = await db.add_medical_literature(literature2)
    doc_id3 = await db.add_medical_literature(literature3)
    
    logger.info('샘플 의학 문헌 추가 완료: %s, %s, %s', doc_id1, doc_id2, doc_id3)


async def add_sample_clinical_guidelines(db):
    """샘플 임상 가이드라인 추가"""
    logger.info('샘플 임상 가이드라인 추가...')
    
    # 샘플 가이드라인 1: 당뇨병 관리 가이드라인
    guideline1 = {
        'title': '제2형 당뇨병 임상 진료 가이드라인 2023',
        'organization': '대한당뇨병학회',
        'publish_date': '2023-01-15',
        'update_date': '2023-01-15',
        'specialty': '내분비학',
        'recommendation_level': 'A',
        'content': """
        # 제2형 당뇨병 임상 진료 가이드라인 2023
        
        ## 진단 기준
        - 공복 혈당 ≥ 126 mg/dL (7.0 mmol/L)
        - 75g 경구당부하검사 후 2시간 혈당 ≥ 200 mg/dL (11.1 mmol/L)
        - 당화혈색소(HbA1c) ≥ 6.5% (48 mmol/mol)
        - 고혈당 증상과 임의 혈당 ≥ 200 mg/dL (11.1 mmol/L)
        
        ## 약물 치료 권고사항
        1. 메트포르민은 금기사항이 없는 한 1차 치료제로 권장됨
        2. 심혈관 질환이 있는 환자는 SGLT2 억제제 또는 GLP-1 수용체 작용제를 조기에 고려
        3. 약물 선택 시 효능, 저혈당 위험, 체중 영향, 부작용, 비용, 환자 선호도를 고려
        
        ## 혈당 조절 목표
        - 일반적으로 당화혈색소 < 6.5-7.0% 목표
        - 합병증 위험이 높거나 고령 환자는 목표를 개별화 (7.0-8.0%)
        
        ## 생활 습관 관리
        - 체중 감량: 과체중/비만 환자는 5-10% 체중 감량 권장
        - 신체 활동: 주 150분 이상 중강도 유산소 운동 및 주 2회 이상 근력운동
        - 식이 요법: 탄수화물, 지방, 단백질의 균형 잡힌 섭취, 과도한 당분 섭취 제한
        
        ## 합병증 선별 검사
        - 망막병증: 매년 안과 검진
        - 신장병증: 매년 미세알부민뇨 및 eGFR 검사
        - 신경병증: 매년 발 검진, 신경병증 증상 평가
        - 심혈관 위험 평가: 정기적 혈압, 혈중 지질 모니터링
        """,
        'metadata': {
            'target_audience': ['내과의사', '가정의학과의사', '내분비내과의사'],
            'evidence_level': '높음',
            'implementation_tools': '환자 교육 자료, 치료 알고리즘, 처방 가이드'
        }
    }
    
    # 샘플 가이드라인 2: 고혈압 관리 가이드라인
    guideline2 = {
        'title': '고혈압 진료 지침 2023',
        'organization': '대한고혈압학회',
        'publish_date': '2023-03-20',
        'update_date': '2023-03-20',
        'specialty': '순환기내과',
        'recommendation_level': 'A',
        'content': """
        # 고혈압 진료 지침 2023
        
        ## 진단 기준
        - 진료실 혈압: 수축기 ≥ 140 mmHg 또는 이완기 ≥ 90 mmHg
        - 자가 혈압: 수축기 ≥ 135 mmHg 또는 이완기 ≥ 85 mmHg
        - 24시간 활동 혈압: 수축기 ≥ 130 mmHg 또는 이완기 ≥ 80 mmHg
        
        ## 치료 목표
        - 일반적으로 수축기 < 140 mmHg 및 이완기 < 90 mmHg
        - 당뇨병, 만성 신장 질환, 관상동맥 질환 환자: 수축기 < 130 mmHg 및 이완기 < 80 mmHg
        - 80세 이상 노인: 개별화된 목표 (수축기 < 150 mmHg)
        
        ## 약물 치료 권고사항
        1. 1차 약제: ACE 억제제, ARB, 칼슘통로차단제, 티아지드계 이뇨제
        2. 2단계 고혈압(160/100 mmHg 이상)은 처음부터 병용요법 고려
        3. 병용요법 시 상승 효과 있는 조합 선택 (레닌-안지오텐신계 차단제 + 칼슘통로차단제 또는 이뇨제)
        
        ## 생활 습관 관리
        - 나트륨 섭취 제한: 하루 6g 미만의 소금 섭취 (2.4g 나트륨)
        - 체중 감량: 과체중/비만 환자는 5-10% 체중 감량 권장
        - 신체 활동: 주 150분 이상 중강도 유산소 운동
        - 금연 및 절주: 완전 금연, 알코올 섭취 제한 (남성 ≤ 2잔/일, 여성 ≤ 1잔/일)
        - 다쉬(DASH) 식이 요법: 과일, 채소, 저지방 유제품 섭취 증가
        
        ## 이차성 고혈압 평가
        - 치료에 저항성이 있거나, 갑자기 발병한 고혈압 또는 비전형적인 특징이 있는 환자에서 이차성 고혈압 평가
        - 신장 동맥 협착, 원발성 알도스테론증, 갈색세포종, 쿠싱 증후군, 수면 무호흡증 등 고려
        """,
        'metadata': {
            'target_audience': ['내과의사', '가정의학과의사', '순환기내과의사'],
            'evidence_level': '높음',
            'implementation_tools': '환자 교육 자료, 위험 평가 도구, 처방 가이드'
        }
    }
    
    # 데이터베이스에 추가
    doc_id1 = await db.add_clinical_guideline(guideline1)
    doc_id2 = await db.add_clinical_guideline(guideline2)
    
    logger.info('샘플 임상 가이드라인 추가 완료: %s, %s', doc_id1, doc_id2)


async def add_sample_patient_records(db):
    """샘플 환자 기록 추가"""
    logger.info('샘플 환자 기록 추가...')
    
    # 샘플 환자 1
    patient1 = {
        'demographics': {
            'age': 52,
            'gender': 'male',
            'height': 175,
            'weight': 82
        },
        'medical_history': {
            'conditions': ['고혈압', '제2형 당뇨병'],
            'allergies': ['페니실린'],
            'surgeries': ['충수돌기염 수술 (2015)'],
            'family_history': ['부: 심장질환', '모: 고혈압']
        },
        'medications': [
            {
                'name': '리시노프릴',
                'dosage': '10mg',
                'frequency': '1일 1회'
            },
            {
                'name': '메트포르민',
                'dosage': '500mg',
                'frequency': '1일 2회'
            }
        ],
        'lab_results': {
            'glucose': 145,
            'HbA1c': 7.2,
            'total_cholesterol': 210,
            'hdl': 45,
            'ldl': 135,
            'triglycerides': 180
        },
        'vitals': {
            'heart_rate': 72,
            'blood_pressure': {
                'systolic': 145,
                'diastolic': 88
            },
            'temperature': 36.5,
            'respiratory_rate': 14,
            'oxygen_saturation': 98
        }
    }
    
    # 샘플 환자 2
    patient2 = {
        'demographics': {
            'age': 68,
            'gender': 'female',
            'height': 160,
            'weight': 70
        },
        'medical_history': {
            'conditions': ['관상동맥 질환', '골관절염', '갑상선 기능 저하증'],
            'allergies': ['설파제', '이부프로펜'],
            'surgeries': ['관상동맥 우회술 (2018)', '오른쪽 무릎 관절 교체 (2020)'],
            'family_history': ['부: 관상동맥 질환', '자매: 유방암']
        },
        'medications': [
            {
                'name': '아스피린',
                'dosage': '81mg',
                'frequency': '1일 1회'
            },
            {
                'name': '아토르바스타틴',
                'dosage': '40mg',
                'frequency': '취침 시'
            },
            {
                'name': '레보티록신',
                'dosage': '75mcg',
                'frequency': '아침 공복'
            }
        ],
        'lab_results': {
            'glucose': 92,
            'HbA1c': 5.6,
            'total_cholesterol': 170,
            'hdl': 60,
            'ldl': 90,
            'triglycerides': 120,
            'tsh': 3.2
        },
        'vitals': {
            'heart_rate': 68,
            'blood_pressure': {
                'systolic': 134,
                'diastolic': 78
            },
            'temperature': 36.7,
            'respiratory_rate': 16,
            'oxygen_saturation': 97
        }
    }
    
    # 샘플 환자 3
    patient3 = {
        'demographics': {
            'age': 35,
            'gender': 'female',
            'height': 165,
            'weight': 58
        },
        'medical_history': {
            'conditions': ['천식', '알레르기성 비염'],
            'allergies': ['꽃가루', '먼지 진드기', '고양이 털'],
            'surgeries': [],
            'family_history': ['모: 천식', '부: 아토피 피부염']
        },
        'medications': [
            {
                'name': '플루티카손/살메테롤 흡입기',
                'dosage': '250/50mcg',
                'frequency': '1일 2회'
            },
            {
                'name': '세티리진',
                'dosage': '10mg',
                'frequency': '필요 시'
            }
        ],
        'lab_results': {
            'glucose': 85,
            'HbA1c': 5.2,
            'total_cholesterol': 165,
            'hdl': 65,
            'ldl': 80,
            'triglycerides': 90,
            'eosinophil_count': '약간 상승'
        },
        'vitals': {
            'heart_rate': 75,
            'blood_pressure': {
                'systolic': 118,
                'diastolic': 72
            },
            'temperature': 36.6,
            'respiratory_rate': 16,
            'oxygen_saturation': 99,
            'peak_flow': '예측치의 85%'
        }
    }
    
    # 데이터베이스에 추가
    patient_id1 = 'patient_001'
    patient_id2 = 'patient_002'
    patient_id3 = 'patient_003'
    
    await db.add_patient_record(patient_id1, patient1)
    await db.add_patient_record(patient_id2, patient2)
    await db.add_patient_record(patient_id3, patient3)
    
    logger.info('샘플 환자 기록 추가 완료: %s, %s, %s', patient_id1, patient_id2, patient_id3)


async def main():
    """메인 함수"""
    try:
        # 테이블 생성
        await create_tables()
        
        # 샘플 데이터 추가
        await add_sample_data()
        
        logger.info('PostgreSQL 데이터베이스 초기화 완료')
    except Exception as e:
        logger.error(f'데이터베이스 초기화 중 오류 발생: {e}', exc_info=True)
        raise


if __name__ == '__main__':
    # 비동기 메인 함수 실행
    asyncio.run(main()) 