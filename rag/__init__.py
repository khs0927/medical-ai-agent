"""
RAG (Retrieval-Augmented Generation) 모듈

RAG 관련 클래스 및 함수를 제공하는 패키지
"""

# 기본 검색기
from rag.mock_retriever import Document, MockRetriever

# PostgreSQL 기반 검색기
try:
    from rag.postgres_retriever import PostgresRetriever
except ImportError:
    PostgresRetriever = None

# Firebase 기반 검색기
try:
    from rag.firebase_retriever import FirebaseRetriever
except ImportError:
    FirebaseRetriever = None

# pgvector 기반 검색기
try:
    from rag.pgvector_retriever import PGVectorRetriever
except ImportError:
    PGVectorRetriever = None

# 웹 기반 검색기
try:
    from rag.web_retriever import WebRetriever
except ImportError:
    WebRetriever = None

__all__ = [
    'Document', 
    'MockRetriever',
    'PostgresRetriever',
    'FirebaseRetriever', 
    'PGVectorRetriever',
    'WebRetriever'
] 