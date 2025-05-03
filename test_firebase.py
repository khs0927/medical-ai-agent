#!/usr/bin/env python3
import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


def test_firebase_connection():
    """Firebase 연결 테스트"""
    print('Firebase 연결 테스트 시작...')
    
    # 서비스 계정 키 파일 경로 출력
    cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    print(f'서비스 계정 키 파일 경로: {cred_path}')
    
    try:
        # Firebase 앱 초기화
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            print('Firebase 앱 초기화 성공')
        
        # Firestore 클라이언트 생성
        db = firestore.client()
        print('Firestore 클라이언트 생성 성공')
        
        # 테스트 컬렉션 확인
        collections = db.collections()
        print('컬렉션 목록:')
        for collection in collections:
            print(f'- {collection.id}')
        
        print('Firebase 연결 테스트 완료!')
        return True
    
    except Exception as e:
        print(f'Firebase 연결 오류: {e}')
        return False

if __name__ == '__main__':
    test_firebase_connection() 