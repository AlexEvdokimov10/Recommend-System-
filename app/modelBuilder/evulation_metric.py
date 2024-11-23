import numpy as np
from typing import List, Any


class EvaluationMetrics:
    def __init__( self,
        accuracy: float,
        precision: float,
        recall: float,
        precision_at_k: float = None,
        recall_at_k: float = None,
        ndcg: float = None,
        mse: float = None,
        rmse: float = None,
        mae: float = None):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.precision_at_k = precision_at_k
        self.recall_at_k = recall_at_k
        self.ndcg = ndcg
        self.mse = mse
        self.rmse = rmse
        self.mae = mae

    @staticmethod
    def calculate_precision_at_k(predictions: List[List[Any]], actuals: List[List[Any]], k: int) -> float:
        precision_scores = []
        for pred, actual in zip(predictions, actuals):
            pred_top_k = set(pred[:k])
            actual_set = set(actual)
            precision_scores.append(len(pred_top_k & actual_set) / k if k > 0 else 0)
        return np.mean(precision_scores)

    @staticmethod
    def calculate_recall_at_k(predictions: List[List[Any]], actuals: List[List[Any]], k: int) -> float:
        recall_scores = []
        for pred, actual in zip(predictions, actuals):
            pred_top_k = set(pred[:k])
            actual_set = set(actual)
            recall_scores.append(len(pred_top_k & actual_set) / len(actual_set) if len(actual_set) > 0 else 0)
        return np.mean(recall_scores)

    @staticmethod
    def calculate_ndcg(predictions: List[List[Any]], actuals: List[List[Any]], k: int) -> float:
        def dcg(relevances):
            return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances[:k]))

        def ideal_dcg(relevances):
            return dcg(sorted(relevances, reverse=True))

        ndcg_scores = []
        for pred, actual in zip(predictions, actuals):
            relevances = [1 if item in actual else 0 for item in pred]
            idcg = ideal_dcg(relevances)
            ndcg_scores.append(dcg(relevances) / idcg if idcg > 0 else 0)
        return np.mean(ndcg_scores)


    @staticmethod
    def calculate_mse(predictions: List[float], actuals: List[float]) -> float:
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        return np.mean((predictions - actuals) ** 2)

    @staticmethod
    def calculate_rmse(predictions: List[float], actuals: List[float]) -> float:
        mse = EvaluationMetrics.calculate_mse(predictions, actuals)
        return np.sqrt(mse)

    @staticmethod
    def calculate_mae(predictions: List[float], actuals: List[float]) -> float:
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        return np.mean(np.abs(predictions - actuals))


    @staticmethod
    def calculate_metrics(predictions: List[Any], actuals: List[Any], k: int = 10) -> 'EvaluationMetrics':
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        accuracy = np.mean(predictions == actuals, dtype=np.float64)
        true_positives = np.sum((predictions == 1) & (actuals == 1))
        precision = true_positives / np.sum(predictions == 1) if np.sum(predictions == 1) > 0 else 0.0
        recall = true_positives / np.sum(actuals == 1) if np.sum(actuals == 1) > 0 else 0.0

        precision_at_k = EvaluationMetrics.calculate_precision_at_k(predictions.tolist(), actuals.tolist(), k)
        recall_at_k = EvaluationMetrics.calculate_recall_at_k(predictions.tolist(), actuals.tolist(), k)
        ndcg = EvaluationMetrics.calculate_ndcg(predictions.tolist(), actuals.tolist(), k)

        mse = EvaluationMetrics.calculate_mse(predictions, actuals)
        rmse = EvaluationMetrics.calculate_rmse(predictions, actuals)
        mae = EvaluationMetrics.calculate_mae(predictions, actuals)

        return EvaluationMetrics(accuracy, precision, recall, precision_at_k, recall_at_k, ndcg, mse, rmse, mae)