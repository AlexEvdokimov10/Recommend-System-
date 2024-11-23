from typing import Dict

import pandas as pd
from .model_parameters import ModelParameters
from .model_strategy import ModelStrategy
from .training_data import TrainingData
from .evulation_metric import EvaluationMetrics

class ModelBuilder:

    def __init__(self, parameters: ModelParameters):
        self.strategy = None
        self.parameters = parameters

    def set_strategy(self, strategy: ModelStrategy):

        self.strategy = strategy

    def build_model(self, data: TrainingData, key_attributes: Dict[str, str]):
        if not self.strategy:
            raise ValueError("Model strategy is not set.")
        print(data.features.columns)
        missing_keys = [key for key in key_attributes.values() if key not in data.features.columns]
        if missing_keys:
            raise ValueError(f"Missing keys in training data: {missing_keys}")
        self.strategy.build(data, self.parameters, key_attributes)
        self.strategy.train(data, self.parameters, key_attributes)

    def tune_model(self, parameters: ModelParameters):

        self.parameters = parameters


class ModelEvaluator:
    def __init__(self, model: ModelStrategy):
        self.model = model

    def evaluate(self, data: pd.DataFrame, labels: pd.DataFrame,key_attributes: Dict[str, str]) -> EvaluationMetrics:
        return self.model.evaluate(data, labels, key_attributes)

    @staticmethod
    def display_metrics(metrics: EvaluationMetrics):
        print(f"Accuracy: {metrics.accuracy:.2f}")
        print(f"Precision: {metrics.precision:.2f}")
        print(f"Recall: {metrics.recall:.2f}")