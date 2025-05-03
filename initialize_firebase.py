#!/usr/bin/env python3
"""
Firebase Firestore 데이터베이스 초기화 스크립트
이 스크립트는 firebase-admin을 사용하여 필요한 Firestore 컬렉션을 생성합니다.
"""
import os
import sys
from typing import Any
from typing import Dict
from typing import List

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# 컬렉션 이름 상수
COLLECTION_MEDICAL_LITERATURE = 'medical_literature'
COLLECTION_CLINICAL_GUIDELINES = 'clinical_guidelines'
COLLECTION_PATIENT_RECORDS = 'patient_records'
COLLECTION_MEDICAL_IMAGES = 'medical_images'

# 샘플 데이터
SAMPLE_MEDICAL_LITERATURE = [
    {
        'title': '고혈압 치료 가이드라인 2025',
        'authors': ['대한고혈압학회'],
        'publication_date': '2025-01-15',
        'journal': '대한고혈압학회지',
        'abstract': '고혈압은 혈압이 지속적으로 140/90 mmHg 이상인 경우로 정의됩니다. '
                    '1단계 고혈압(140-159/90-99 mmHg)의 경우 생활습관 개선과 함께 약물 치료를 시작할 수 있으며, '
                    '2단계 고혈압(≥160/100 mmHg)은 즉시 약물 치료를 시작해야 합니다. '
                    '일차 약제로는 ACE 억제제, ARB, 칼슘 채널 차단제, 티아지드계 이뇨제가 권장됩니다.',
        'doi': '10.5551/jkhs.2025.01.001',
        'mesh_terms': ['고혈압', '치료', '가이드라인']
    },
    {
        'title': '당뇨병 관리: 최신 연구 동향',
        'authors': ['김영희', '이철수', '박지민'],
        'publication_date': '2024-08-10',
        'journal': '당뇨병학회지',
        'abstract': '제2형 당뇨병 환자의 경우 메트포르민이 일반적으로 1차 치료제로 사용됩니다. '
                    '혈당 조절이 충분하지 않을 경우 SGLT-2 억제제나 GLP-1 수용체 작용제를 추가할 수 있으며, '
                    '특히 심혈관질환 위험이 높은 환자에게 효과적입니다. '
                    '최근 연구에 따르면 일부 환자에서 저탄수화물 식이요법이 인슐린 감수성을 개선할 수 있습니다.',
        'doi': '10.5551/jkds.2024.08.005',
        'mesh_terms': ['당뇨병', '치료', '식이요법']
    }
]

SAMPLE_CLINICAL_GUIDELINES = [
    {
        'title': '뇌졸중 진단 및 치료 가이드라인',
        'organization': '대한신경과학회',
        'publish_date': '2024-05-20',
        'update_date': '2024-05-20',
        'specialty': '신경과',
        'recommendation_level': 'A',
        'content': '급성 허혈성 뇌졸중의 경우, 증상 발생 4.5시간 이내에 정맥내 혈전용해술(tPA)을 고려해야 합니다. '
                   '단, 출혈성 뇌졸중이 배제되어야 하며, 금기사항이 없어야 합니다. '
                   '6시간 이내의 대혈관 폐색 환자에서는 기계적 혈전제거술이 권장됩니다.'
    },
    {
        'title': '알레르기 비염 치료 가이드라인',
        'organization': '대한이비인후과학회',
        'publish_date': '2023-09-15',
        'update_date': '2024-03-10',
        'specialty': '이비인후과',
        'recommendation_level': 'B',
        'content': '계절성 알레르기 비염의 경우, 비강 내 스테로이드 스프레이와 2세대 항히스타민제의 병용 요법이 권장됩니다. '
                   '증상이 심한 경우 단기간 경구 스테로이드를 고려할 수 있습니다. '
                   '장기적인 관리에는 알레르겐 회피와 면역치료가 효과적입니다.'
    }
]

SAMPLE_PATIENT_RECORDS = [
    {
        'patient_id': 'patient001',
        'demographics': {
            'age': 45,
            'gender': 'male',
            'height': 175,
            'weight': 78
        },
        'medical_history': {
            'conditions': ['고혈압', '제2형 당뇨병'],
            'allergies': ['페니실린'],
            'surgeries': ['충수돌기염 수술 (2010)'],
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
        'vitals': {
            'heart_rate': 72,
            'blood_pressure': {
                'systolic': 135,
                'diastolic': 85
            },
            'oxygen_level': 98,
            'temperature': 36.5
        }
    },
    {
        'patient_id': 'patient002',
        'demographics': {
            'age': 62,
            'gender': 'female',
            'height': 162,
            'weight': 65
        },
        'medical_history': {
            'conditions': ['골관절염', '고지혈증'],
            'allergies': ['설파제'],
            'surgeries': ['무릎 인공관절 수술 (2022)'],
            'family_history': ['모: 유방암', '자매: 류마티스 관절염']
        },
        'medications': [
            {
                'name': '아토르바스타틴',
                'dosage': '20mg',
                'frequency': '1일 1회'
            },
            {
                'name': '아세트아미노펜',
                'dosage': '500mg',
                'frequency': '필요시'
            }
        ],
        'vitals': {
            'heart_rate': 78,
            'blood_pressure': {
                'systolic': 142,
                'diastolic': 88
            },
            'oxygen_level': 97,
            'temperature': 36.7
        }
    }
]

def initialize_firebase() -> firestore.Client:
    """Firebase 앱 초기화 및 Firestore 클라이언트 반환"""
    print('Firebase 초기화 중...')
    
    # 서비스 계정 키 파일 경로
    cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not cred_path or not os.path.exists(cred_path):
        print(f'오류: 서비스 계정 키 파일을 찾을 수 없습니다. ({cred_path})')
        print('GOOGLE_APPLICATION_CREDENTIALS 환경 변수를 설정했는지 확인하세요.')
        sys.exit(1)
    
    print(f'서비스 계정 키 파일: {cred_path}')
    
    # Firebase 앱 초기화
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        
        # Firestore 클라이언트 생성
        db = firestore.client()
        print('Firebase 초기화 성공')
        return db
    
    except Exception as e:
        print(f'Firebase 초기화 오류: {e}')
        print('\n참고: Firestore API를 활성화해야 합니다.')
        print('https://console.developers.google.com/apis/api/firestore.googleapis.com/overview?project=medical-ai-agent 에서 API를 활성화하세요.')
        sys.exit(1)

def create_collection_with_samples(db: firestore.Client, collection_name: str, samples: List[Dict[str, Any]]) -> None:
    """지정된 컬렉션에 샘플 데이터 추가"""
    print(f'\n\'{collection_name}\' 컬렉션에 샘플 데이터 추가 중...')
    
    collection_ref = db.collection(collection_name)
    
    for sample in samples:
        # 문서 ID가 지정된 경우 (예: 환자 기록)
        if collection_name == COLLECTION_PATIENT_RECORDS and 'patient_id' in sample:
            doc_id = sample['patient_id']
            doc_ref = collection_ref.document(doc_id)
        else:
            # 자동 ID 생성
            doc_ref = collection_ref.document()
        
        # 타임스탬프 추가
        sample['created_at'] = firestore.SERVER_TIMESTAMP
        sample['updated_at'] = firestore.SERVER_TIMESTAMP
        
        # 문서 추가
        doc_ref.set(sample)
        print(f'- 문서 추가됨: {doc_ref.id}')
    
    print(f'\'{collection_name}\' 컬렉션에 {len(samples)}개 문서 추가 완료')

def main():
    """메인 함수"""
    print('=' * 50)
    print(' Firebase Firestore 데이터베이스 초기화')
    print('=' * 50)
    
    # Firebase 초기화
    db = initialize_firebase()
    
    # 각 컬렉션에 샘플 데이터 추가
    create_collection_with_samples(db, COLLECTION_MEDICAL_LITERATURE, SAMPLE_MEDICAL_LITERATURE)
    create_collection_with_samples(db, COLLECTION_CLINICAL_GUIDELINES, SAMPLE_CLINICAL_GUIDELINES)
    create_collection_with_samples(db, COLLECTION_PATIENT_RECORDS, SAMPLE_PATIENT_RECORDS)
    
    print('\nFirebase Firestore 데이터베이스 초기화 완료!')
    print('다음 단계: \'python run_server.py\'를 실행하여 서버를 시작하세요.')

if __name__ == '__main__':
    main() 