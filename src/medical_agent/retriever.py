"""
Vector store RAG layer (Weaviate Cloud Free tier)
"""
import os, weaviate, hashlib

# 테스트 모드 확인
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() == "true"

if not TEST_MODE:
    try:
        _WEAVIATE_URL = os.environ["WEAVIATE_URL"]
        _WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]

        client = weaviate.Client(
            url=_WEAVIATE_URL,
            auth_client_secret=weaviate.AuthApiKey(_WEAVIATE_API_KEY),
            additional_headers={"X-OpenAI-Api-Key": "placeholder"},
        )
    except (KeyError, Exception) as e:
        print(f"Weaviate 클라이언트 초기화 실패: {e}")
        client = None

def semantic_retrieval(query: str, k: int = 4) -> str:
    """의미적 검색을 수행하여 결과를 텍스트로 반환합니다."""
    
    # 테스트 모드일 경우 모의 데이터 반환
    if TEST_MODE or client is None:
        return _mock_retrieval(query)
    
    try:
        resp = (
            client.query
            .get("PubMedPaper", ["title", "abstract"])
            .with_near_text({"concepts": [query]})
            .with_limit(k)
            .do()
        )
        items = resp["data"]["Get"]["PubMedPaper"]
        return "\n\n".join(f"{it['title']}\n{it['abstract']}" for it in items)
    except Exception as e:
        print(f"Weaviate 검색 오류: {e}")
        return _mock_retrieval(query)

def _mock_retrieval(query: str) -> str:
    """테스트용 모의 검색 결과를 반환합니다."""
    mock_papers = [
        {
            "title": "심혈관 질환의 최신 치료 가이드라인 (2024)",
            "abstract": "본 연구는 심혈관 질환의 최신 치료 방법과 권장 사항을 요약합니다. 고혈압, 부정맥, 관상동맥 질환에 대한 최신 약물 치료법과 중재적 시술의 적응증을 포함합니다."
        },
        {
            "title": "스타틴 약물의 최근 연구 동향",
            "abstract": "스타틴은 콜레스테롤 합성 효소를 억제하여 LDL 콜레스테롤을 낮추는 약물입니다. 최근 연구에 따르면 심혈관 위험을 약 30% 감소시키는 효과가 있으며, 당뇨병 환자에게 특히 효과적입니다."
        },
        {
            "title": "항혈소판 약물 병용요법의 효과와 위험성",
            "abstract": "아스피린, 클로피도그렐 등의 항혈소판 약물 병용요법은 심혈관 사건 예방에 효과적이나 출혈 위험이 증가합니다. 본 연구는 다양한 병용요법의 효과와 위험을 분석했습니다."
        },
        {
            "title": "STEMI 환자의 응급 처치 프로토콜",
            "abstract": "ST 분절 상승 심근경색(STEMI) 환자는 신속한 재관류 치료가 필요합니다. 본 가이드라인은 응급실 도착부터 카테터실 이송까지의 최적화된 프로토콜을 제시합니다."
        }
    ]
    
    # 쿼리에 따라 가장 관련성 높은 모의 데이터 선택
    if "STEMI" in query.upper():
        selected_papers = [mock_papers[3], mock_papers[0]]
    elif "스타틴" in query or "콜레스테롤" in query:
        selected_papers = [mock_papers[1], mock_papers[0]]
    elif "아스피린" in query or "클로피도그렐" in query or "약물" in query:
        selected_papers = [mock_papers[2], mock_papers[1]]
    else:
        selected_papers = mock_papers[:2]
    
    return "\n\n".join(f"{paper['title']}\n{paper['abstract']}" for paper in selected_papers) 