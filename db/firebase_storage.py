"""
Firebase Storage 파일 관리 유틸리티
"""
import logging
import os
from typing import List
from typing import Optional

from firebase_admin import storage

from db.firebase_config import initialize_firebase

# 로깅 설정
logger = logging.getLogger(__name__)


class FirebaseStorage:
    """Firebase Storage를 사용한 파일 관리"""

    def __init__(self, bucket_name: Optional[str] = None):
        """Firebase Storage 초기화
        
        Args:
            bucket_name: 사용할 버킷 이름(없으면 기본 버킷 사용)
        """
        # Firebase 앱 초기화 (필요한 경우)
        initialize_firebase()
        
        # 버킷 초기화
        self.bucket_name = bucket_name
        self.bucket = storage.bucket(bucket_name)
        
        logger.info(f'Firebase Storage 초기화 완료 (버킷: {bucket_name or \'기본 버킷\'})')
    
    async def upload_file(self, local_path: str, remote_path: str) -> str:
        """파일을 Firebase Storage에 업로드
        
        Args:
            local_path: 업로드할 로컬 파일 경로
            remote_path: 저장할 원격 경로
            
        Returns:
            업로드된 파일의 공개 URL
        """
        try:
            if not os.path.exists(local_path):
                logger.error(f'업로드할 파일이 존재하지 않습니다: {local_path}')
                raise FileNotFoundError(f'File not found: {local_path}')
            
            # 파일 업로드
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            
            # 공개 URL 생성
            blob.make_public()
            public_url = blob.public_url
            
            logger.info(f'파일 업로드 완료: {local_path} -> {remote_path}')
            return public_url
            
        except Exception as e:
            logger.error(f'파일 업로드 중 오류 발생: {e}', exc_info=True)
            raise
    
    async def download_file(self, remote_path: str, local_path: str) -> str:
        """Firebase Storage에서 파일 다운로드
        
        Args:
            remote_path: 다운로드할 원격 파일 경로
            local_path: 저장할 로컬 경로
            
        Returns:
            다운로드된 파일의 로컬 경로
        """
        try:
            # 로컬 디렉토리 생성 (필요한 경우)
            local_dir = os.path.dirname(local_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir)
            
            # 파일 다운로드
            blob = self.bucket.blob(remote_path)
            blob.download_to_filename(local_path)
            
            logger.info(f'파일 다운로드 완료: {remote_path} -> {local_path}')
            return local_path
            
        except Exception as e:
            logger.error(f'파일 다운로드 중 오류 발생: {e}', exc_info=True)
            raise
    
    async def delete_file(self, remote_path: str) -> bool:
        """Firebase Storage에서 파일 삭제
        
        Args:
            remote_path: 삭제할 파일 경로
            
        Returns:
            삭제 성공 여부
        """
        try:
            # 파일 삭제
            blob = self.bucket.blob(remote_path)
            blob.delete()
            
            logger.info(f'파일 삭제 완료: {remote_path}')
            return True
            
        except Exception as e:
            logger.error(f'파일 삭제 중 오류 발생: {e}', exc_info=True)
            return False
    
    async def list_files(self, prefix: str = '', delimiter: str = '/') -> List[str]:
        """디렉토리 내 파일 목록 조회
        
        Args:
            prefix: 조회할 디렉토리 경로
            delimiter: 경로 구분자
            
        Returns:
            파일 경로 목록
        """
        try:
            blobs = self.bucket.list_blobs(prefix=prefix, delimiter=delimiter)
            file_paths = [blob.name for blob in blobs]
            
            return file_paths
            
        except Exception as e:
            logger.error(f'파일 목록 조회 중 오류 발생: {e}', exc_info=True)
            return []
    
    async def generate_signed_url(self, remote_path: str, expiration: int = 3600) -> str:
        """서명된 URL 생성 (임시 접근용)
        
        Args:
            remote_path: 파일 경로
            expiration: 만료 시간(초, 기본 1시간)
            
        Returns:
            서명된 URL
        """
        try:
            import datetime
            
            blob = self.bucket.blob(remote_path)
            url = blob.generate_signed_url(
                expiration=datetime.timedelta(seconds=expiration),
                method='GET'
            )
            
            logger.info(f'서명된 URL 생성 완료: {remote_path} (만료: {expiration}초)')
            return url
            
        except Exception as e:
            logger.error(f'서명된 URL 생성 중 오류 발생: {e}', exc_info=True)
            raise 