"""
RAG 검색기를 위한 기본 인터페이스 정의
"""

import abc
from typing import Any, Dict, List, Optional


class BaseRetriever(abc.ABC):
    """RAG 검색기를 위한 추상 기본 클래스"""

    @abc.abstractmethod
    async def retrieve_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """쿼리와 관련된 문서 검색

        Args:
            query: 검색 쿼리
            limit: 반환할 최대 문서 수

        Returns:
            관련 문서 목록
        """
        pass

    @abc.abstractmethod
    async def add_document(self, document: Dict[str, Any]) -> str:
        """문서 추가

        Args:
            document: 추가할 문서

        Returns:
            생성된 문서 ID
        """
        pass

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """문서 가져오기

        Args:
            doc_id: 문서 ID

        Returns:
            문서 데이터 또는 None (존재하지 않는 경우)
        """
        raise NotImplementedError('This retriever does not implement get_document')

    async def update_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """문서 업데이트

        Args:
            doc_id: 문서 ID
            document: 업데이트할 문서 데이터

        Returns:
            성공 여부
        """
        raise NotImplementedError('This retriever does not implement update_document')

    async def delete_document(self, doc_id: str) -> bool:
        """문서 삭제

        Args:
            doc_id: 문서 ID

        Returns:
            성공 여부
        """
        raise NotImplementedError('This retriever does not implement delete_document')

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """키워드 기반 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 문서 수

        Returns:
            관련 문서 목록
        """
        return await self.retrieve_documents(query, limit=top_k)

    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """의미론적 검색 (임베딩 기반)

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 문서 수

        Returns:
            관련 문서 목록
        """
        # 기본적으로 일반 검색으로 폴백, 자식 클래스에서 재정의 가능
        return await self.search(query, top_k=top_k) 