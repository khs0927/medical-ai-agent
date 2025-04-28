from __future__ import annotations

"""
ECG 분석 컴포넌트 (Phase 3)

심전도(ECG) 신호 분석을 위한 기본 클래스 구현 
실제 모델은 포함하지 않고 모의 데이터 반환
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ECGAnalysisResult:
    """ECG 분석 결과 클래스"""
    predictions: Dict[str, float]  # 각 상태별 확률
    abnormality_regions: List[Dict[str, Any]]  # 이상 구간 정보
    interpretation: str  # 텍스트 해석


class ECGAnalyzer:
    """ECG 분석기 (MVP 모의 구현)"""

    def __init__(self, model_path: Optional[str] = None):
        """초기화"""
        self.model_path = model_path
        self.label_map = self._load_label_map()
        logger.info("ECG 분석기 초기화 완료")

    def _load_label_map(self) -> Dict[int, str]:
        """레이블 매핑 로드"""
        return {
            0: "Normal",
            1: "Atrial Fibrillation",
            2: "First-degree AV Block",
            3: "Left Bundle Branch Block",
            4: "Right Bundle Branch Block",
            5: "Premature Atrial Contraction",
            6: "Premature Ventricular Contraction",
            7: "ST-segment Depression",
            8: "ST-segment Elevation"
        }

    def preprocess_ecg(self, signal: np.ndarray) -> np.ndarray:
        """ECG 신호 전처리"""
        # MVP에서는 간단한 전처리만 수행
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)
        
        # 신호 정규화 (표준화)
        if signal.size > 0:  # 빈 배열이 아닌 경우
            signal_normalized = (signal - np.mean(signal)) / np.std(signal)
        else:
            signal_normalized = signal
            
        return signal_normalized

    def _resample_signal(self, signal: np.ndarray, target_freq: int = 250) -> np.ndarray:
        """신호 리샘플링 (실제 구현 시 필요)"""
        # MVP에서는 리샘플링 없이 원 신호 반환
        return signal
    
    def _apply_bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """대역통과 필터 적용 (실제 구현 시 필요)"""
        # MVP에서는 필터링 없이 원 신호 반환
        return signal

    def _detect_abnormal_regions(self, signal: np.ndarray, predictions: Dict[str, float]) -> List[Dict[str, Any]]:
        """이상 구간 탐지 (모의 구현)"""
        # 실제로는 각 상태별 특성 패턴을 찾는 알고리즘 구현 필요
        # MVP에서는 모의 데이터 반환
        abnormal_regions = []
        
        for label, prob in predictions.items():
            if label != "Normal" and prob > 0.5:
                # 실제 구간은 신호 분석이 필요하지만 모의 데이터로 대체
                if signal.size >= 1000:  # 충분한 신호 길이가 있을 경우
                    start_sample = np.random.randint(0, signal.shape[1] - 250)
                    length = np.random.randint(100, 250)
                    end_sample = min(start_sample + length, signal.shape[1] - 1)
                    
                    # 250Hz 샘플링 가정, 시간으로 변환
                    start_time = start_sample / 250.0
                    end_time = end_sample / 250.0
                    
                    abnormal_regions.append({
                        "type": label,
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "confidence": prob
                    })
                else:
                    # 짧은 신호의 경우 간단한 모의 데이터
                    abnormal_regions.append({
                        "type": label,
                        "start_time": 0.5,
                        "end_time": 1.5,
                        "start_sample": 125,
                        "end_sample": 375,
                        "confidence": prob
                    })
        
        return abnormal_regions
    
    def _generate_interpretation(self, predictions: Dict[str, float], abnormal_regions: List[Dict[str, Any]]) -> str:
        """예측 결과 텍스트 해석 생성"""
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        highest_pred = sorted_preds[0]
        
        if highest_pred[0] == "Normal" and highest_pred[1] > 0.7:
            return f"심전도는 정상으로 판단됩니다 (신뢰도: {highest_pred[1]:.2f})."
        
        # 의미있는 이상 유형 추출
        significant_abnormalities = [p for p in sorted_preds if p[0] != "Normal" and p[1] > 0.5]
        
        if not significant_abnormalities:
            return "명확한 심전도 이상이 탐지되지 않았습니다. 일부 경계선상 변화가 있을 수 있습니다."
        
        # 주요 이상 유형 설명
        primary_abnormality = significant_abnormalities[0]
        
        description = f"주요 심전도 이상 발견: {primary_abnormality[0]} (신뢰도: {primary_abnormality[1]:.2f})."
        
        if len(significant_abnormalities) > 1:
            secondary = significant_abnormalities[1]
            description += f" 추가 발견된 이상: {secondary[0]} (신뢰도: {secondary[1]:.2f})."
        
        # 이상 구간 정보 추가
        if abnormal_regions:
            region_desc = abnormal_regions[0]["type"]
            region_time = f"{abnormal_regions[0]['start_time']:.1f}초-{abnormal_regions[0]['end_time']:.1f}초"
            description += f" 이상 구간: {region_desc} ({region_time})."
        
        return description

    def analyze(self, ecg_signal: np.ndarray) -> ECGAnalysisResult:
        """ECG 신호 분석"""
        try:
            logger.info("ECG 신호 분석 시작")
            
            # 신호 형상 확인
            if not isinstance(ecg_signal, np.ndarray):
                logger.warning("입력이 NumPy 배열이 아닙니다. 변환을 시도합니다.")
                ecg_signal = np.array(ecg_signal)
            
            # 전처리
            signal_normalized = self.preprocess_ecg(ecg_signal)
            
            # 실제 모델이면 예측 실행
            # MVP에서는 모의 예측 데이터 반환
            predictions = self._mock_predictions()
            
            # 이상 구간 탐지
            abnormality_regions = self._detect_abnormal_regions(signal_normalized, predictions)
            
            # 텍스트 해석 생성
            interpretation = self._generate_interpretation(predictions, abnormality_regions)
            
            # 결과 반환
            result = ECGAnalysisResult(
                predictions=predictions,
                abnormality_regions=abnormality_regions,
                interpretation=interpretation
            )
            
            logger.info("ECG 분석 완료")
            return result
            
        except Exception as e:
            logger.error(f"ECG 분석 중 오류 발생: {e}", exc_info=True)
            # 오류 발생 시 기본 결과 반환
            return ECGAnalysisResult(
                predictions={"Error": 1.0},
                abnormality_regions=[],
                interpretation=f"분석 중 오류가 발생했습니다: {str(e)}"
            )
    
    def _mock_predictions(self) -> Dict[str, float]:
        """모의 예측 결과 생성 (MVP용)"""
        # 무작위로 정상 또는 이상 상태 생성
        is_normal = np.random.random() > 0.6
        
        if is_normal:
            # 정상 심전도
            predictions = {
                "Normal": np.random.uniform(0.8, 0.98),
                "Atrial Fibrillation": np.random.uniform(0.01, 0.1),
                "Premature Atrial Contraction": np.random.uniform(0.01, 0.1),
                "ST-segment Depression": np.random.uniform(0.01, 0.05)
            }
        else:
            # 비정상 심전도 - 한 가지 이상 유형 선택
            abnormal_type = np.random.choice([
                "Atrial Fibrillation", 
                "First-degree AV Block",
                "Premature Ventricular Contraction",
                "ST-segment Elevation"
            ])
            
            # 기본값 설정
            predictions = {
                "Normal": np.random.uniform(0.05, 0.3),
                "Atrial Fibrillation": np.random.uniform(0.01, 0.1),
                "First-degree AV Block": np.random.uniform(0.01, 0.1),
                "Premature Ventricular Contraction": np.random.uniform(0.01, 0.1),
                "ST-segment Elevation": np.random.uniform(0.01, 0.1)
            }
            
            # 선택된 이상 유형에 높은 확률 부여
            predictions[abnormal_type] = np.random.uniform(0.7, 0.95)
            
        # 나머지 레이블에 낮은 확률 할당
        for label in self.label_map.values():
            if label not in predictions:
                predictions[label] = np.random.uniform(0.001, 0.05)
                
        # 확률 합이 1이 되도록 정규화
        total = sum(predictions.values())
        normalized_predictions = {k: v/total for k, v in predictions.items()}
        
        return normalized_predictions 