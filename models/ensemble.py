from __future__ import annotations

"""Lightweight ensemble wrapper (Phase 1 MVP)

현재 프로젝트에는 `agent.py` 의 `MedicalAgent` 가 Gemini·MedLLaMA 하이브리드 호출 기능을 보유하고 있다.
MVP 단계에서는 별도 복잡한 가중 투표 대신 MedicalAgent 의 내부 로직을 그대로 활용하여
단일 응답을 반환하도록 한다. 추후 Phase 2 이후 실제 복수 모델과 가중치 로직을 여기에 확장할 예정이다.

Example:
    >>> from models.ensemble import MedicalEnsemble
    >>> ensemble = MedicalEnsemble()
    >>> result = ensemble.predict("고혈압 치료 방법은?" )
    >>> print(result["response"])
"""

from typing import Any, Dict, Optional
import logging

from agent import MedicalAgent

logger = logging.getLogger(__name__)


class MedicalEnsemble:
    """경량 앙상블 클래스 (Gemini + MedLLaMA)"""

    def __init__(self, models_config: Optional[list[dict[str, Any]]] = None):
        # 현재는 config 무시하고 MedicalAgent 하나만 사용
        self.agent = MedicalAgent()
        ok = self.agent.initialize_llm_client()
        if not ok:
            logger.warning("LLM 초기화 실패 – MedicalEnsemble 응답 시 mock 데이터만 반환될 수 있습니다.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """LLM 예측 실행

        Args:
            prompt: 최종 프롬프트 텍스트
            context: (MVP) 미사용 – 추후 RAG, 환자 정보 등 전달용
        Returns:
            dict: {"response": str, "confidence": float}
        """
        logger.info("MedicalEnsemble.predict 호출")
        try:
            answer = self.agent._call_llm(prompt)  # pylint: disable=protected-access
        except Exception as exc:  # noqa: BLE001
            logger.error("Ensemble 내부 오류: %s", exc, exc_info=True)
            answer = None

        if not answer:
            # fallback – MedicalAgent mock 또는 빈 응답
            answer = "죄송합니다. 현재 답변을 생성할 수 없습니다."
            confidence = 0.0
        else:
            confidence = 0.9  # MVP 임시 고정값

        return {
            "response": answer,
            "confidence": confidence,
        } 