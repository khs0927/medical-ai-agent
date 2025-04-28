from __future__ import annotations

"""
의료 AI 평가 모듈 (Phase 5)

모델 응답 평가, 데이터셋 평가 등을 위한 간단한 평가 기능 구현
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class MedicalEvaluator:
    """MVP 단계 의료 AI 평가 도구"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """평가기 초기화"""
        self.config = config or {}
        self.output_dir = self.config.get("output_dir", "evaluation_results")
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 로그 설정
        self._setup_logger()
        
        # 평가 결과 저장
        self.evaluation_results = []

    def _setup_logger(self):
        """평가 로거 설정"""
        # 파일 핸들러
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/evaluation.log")
        file_handler.setLevel(logging.INFO)
        
        # 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    def _load_test_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """테스트 데이터셋 로드"""
        logger.info(f"데이터셋 로드: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            logger.error(f"데이터셋을 찾을 수 없음: {dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # 파일 확장자 확인
        _, ext = os.path.splitext(dataset_path)
        
        if ext.lower() == '.json':
            # JSON 형식 로드
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                
            # 단일 객체가 아닌 리스트인지 확인
            if not isinstance(dataset, list):
                dataset = [dataset]
                
        elif ext.lower() in ['.csv', '.tsv']:
            # CSV/TSV 형식 로드
            sep = ',' if ext.lower() == '.csv' else '\t'
            df = pd.read_csv(dataset_path, sep=sep)
            dataset = df.to_dict('records')
            
        else:
            logger.error(f"지원하지 않는 파일 형식: {ext}")
            raise ValueError(f"Unsupported file format: {ext}")
        
        logger.info(f"데이터셋 로드 완료: {len(dataset)} 항목")
        return dataset

    async def evaluate_model(self, model_name: str, test_dataset: str) -> Dict[str, Any]:
        """모델 성능 평가"""
        logger.info(f"모델 평가 시작: {model_name}, 데이터셋: {test_dataset}")
        
        # 테스트 데이터셋 로드
        test_data = self._load_test_dataset(test_dataset)
        
        # 모델 예측 실행
        predictions = await self._run_model_predictions(model_name, test_data)
        
        # 성능 지표 계산
        metrics = self._calculate_metrics(test_data, predictions)
        
        # 결과 저장
        timestamp = datetime.now()
        evaluation_result = {
            "model_name": model_name,
            "dataset": test_dataset,
            "timestamp": timestamp.isoformat(),
            "dataset_size": len(test_data),
            "metrics": metrics,
            "config": self.config
        }
        
        self.evaluation_results.append(evaluation_result)
        self._save_results(evaluation_result)
        
        logger.info(f"평가 완료: {model_name}, 정확도: {metrics.get('accuracy', 0):.4f}")
        
        return evaluation_result

    async def _run_model_predictions(self, model_name: str, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """모델 예측 실행"""
        # 이 함수는 모델 이름에 따라 적절한 모델 호출 로직 구현 필요
        # 지금은 의존성을 줄이기 위해 에뮬레이션으로 구현
        
        from models.ensemble import MedicalEnsemble
        
        try:
            # 모델 초기화
            ensemble = MedicalEnsemble()
            
            # 각 테스트 샘플에 대한 예측 실행
            predictions = []
            
            for i, sample in enumerate(test_data):
                try:
                    if i % 10 == 0:
                        logger.info(f"샘플 처리 중: {i+1}/{len(test_data)}")
                    
                    # 필요한 필드 확인
                    if "query" not in sample:
                        raise ValueError(f"샘플에 'query' 필드가 없습니다: {sample}")
                    
                    # 모델 예측
                    query = sample["query"]
                    context = sample.get("context", {})
                    result = ensemble.predict(query, context=context)
                    
                    # 예측 결과와 실제 값 비교
                    predictions.append({
                        "sample_id": sample.get("id", i),
                        "query": query,
                        "prediction": result["response"],
                        "ground_truth": sample.get("answer", sample.get("ground_truth", "")),
                        "confidence": result["confidence"]
                    })
                    
                except Exception as e:
                    logger.error(f"샘플 {i} 처리 중 오류: {e}")
                    predictions.append({
                        "sample_id": sample.get("id", i),
                        "query": sample.get("query", ""),
                        "prediction": "ERROR",
                        "ground_truth": sample.get("answer", sample.get("ground_truth", "")),
                        "confidence": 0.0,
                        "error": str(e)
                    })
            
            return predictions
            
        except Exception as e:
            logger.error(f"예측 실행 중 오류: {e}", exc_info=True)
            # 빈 예측 반환
            return [{"error": str(e)}] * len(test_data)

    def _calculate_metrics(self, test_data: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """성능 지표 계산"""
        metrics = {}
        
        # 기본 통계
        valid_predictions = [p for p in predictions if "error" not in p]
        metrics["valid_ratio"] = len(valid_predictions) / len(predictions) if predictions else 0
        
        # 분류 문제인지 질문-응답 문제인지 확인
        if all("label" in sample for sample in test_data):
            # 분류 지표 계산
            if SKLEARN_AVAILABLE:  
                y_true = [sample.get("label", "") for sample in test_data]
                y_pred = [pred.get("prediction", "") for pred in predictions]
                
                metrics["accuracy"] = accuracy_score(y_true, y_pred)
                metrics["precision_macro"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                metrics["recall_macro"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                metrics["f1_macro"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            else:
                # sklearn 없이 정확도 계산
                correct = sum(1 for i, sample in enumerate(test_data) if
                             i < len(predictions) and sample.get("label", "") == predictions[i].get("prediction", ""))
                metrics["accuracy"] = correct / len(test_data) if test_data else 0
        else:
            # 생성형 모델 평가 (간단한 매칭)
            exact_match = 0
            for i, pred in enumerate(predictions):
                if i < len(test_data):
                    gt = test_data[i].get("answer", test_data[i].get("ground_truth", ""))
                    prediction = pred.get("prediction", "")
                    
                    # 정확히 일치하는지 확인 (대소문자 무시)
                    if gt.strip().lower() == prediction.strip().lower():
                        exact_match += 1
            
            metrics["exact_match_ratio"] = exact_match / len(test_data) if test_data else 0
            
            # 평균 신뢰도
            avg_confidence = sum(p.get("confidence", 0) for p in predictions) / len(predictions) if predictions else 0
            metrics["avg_confidence"] = avg_confidence
        
        return metrics

    def _save_results(self, result: Dict[str, Any]) -> None:
        """평가 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = result["model_name"].replace(" ", "_")
        dataset_name = os.path.basename(result["dataset"]).replace(".", "_")
        
        filename = f"{model_name}_{dataset_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"평가 결과 저장 완료: {filepath}")

    def generate_report(self, format: str = "text") -> str:
        """평가 결과 보고서 생성"""
        if not self.evaluation_results:
            return "평가 결과가 없습니다."
        
        if format.lower() == "text":
            # 텍스트 형식 보고서
            report = []
            report.append("=" * 50)
            report.append("의료 AI 모델 평가 보고서")
            report.append("=" * 50)
            
            for result in self.evaluation_results:
                report.append(f"\n모델: {result['model_name']}")
                report.append(f"데이터셋: {result['dataset']}")
                report.append(f"평가 시간: {result['timestamp']}")
                report.append(f"데이터셋 크기: {result['dataset_size']}")
                report.append("\n성능 지표:")
                
                metrics = result["metrics"]
                for metric_name, metric_value in metrics.items():
                    report.append(f"  {metric_name}: {metric_value:.4f}")
                
                report.append("-" * 50)
            
            return "\n".join(report)
            
        elif format.lower() == "json":
            # JSON 형식 보고서
            return json.dumps(self.evaluation_results, indent=2, ensure_ascii=False)
            
        else:
            return f"지원하지 않는 보고서 형식: {format}" 