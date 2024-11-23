import pandas as pd

class TrainingData:

    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame):
        if features.empty or labels.empty:
            raise ValueError("Features and labels must not be empty.")
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same number of rows.")

        self.features = features
        self.labels = labels

    def get_features(self) -> pd.DataFrame:
        return self.features

    def get_labels(self) -> pd.DataFrame:
        return self.labels

    def describe(self) -> str:
        return (
            f"Features:\n{self.features.info()}\n\n"
            f"Labels:\n{self.labels.info()}\n"
        )