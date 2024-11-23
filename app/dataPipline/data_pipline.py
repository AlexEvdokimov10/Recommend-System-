from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as ks
from tf_keras.layers import Dense
from tf_keras.models import Sequential


class PipelineStep(ABC):
    @abstractmethod
    def process(self, context):
        pass


class PipelineContext:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.processed_data = raw_data
        self.metadata = {}

    def set_processed_data(self, data):
        self.processed_data = data

    def get_processed_data(self):
        return self.processed_data


class PipelineLogger:
    def __init__(self):
        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def display_logs(self):
        for log in self.logs:
            print(log)


class DataPipeline:
    def __init__(self):
        self.steps = []
        self.logger = PipelineLogger()

    def add_step(self, step):
        self.steps.append(step)

    def execute(self, context):
        for step in self.steps:
            try:
                self.logger.log(f"Executing {step.__class__.__name__}")
                data = context.get_processed_data()
                self.logger.log(
                    f"Data shape before {step.__class__.__name__}: {data.shape if hasattr(data, 'shape') else 'Unknown'}")
                step.process(context)
                data = context.get_processed_data()
                self.logger.log(
                    f"Data shape after {step.__class__.__name__}: {data.shape if hasattr(data, 'shape') else 'Unknown'}")
                self.logger.log(f"Completed {step.__class__.__name__}")
            except Exception as e:
                self.logger.log(f"Error in step {step.__class__.__name__}: {e}")
                raise e

    def remove_step(self, step):
        self.steps.remove(step)


class BasicDataCleaningStep(PipelineStep):
    def process(self, context):
        data = context.get_processed_data()

        if isinstance(data, pd.DataFrame):
            print(f"Original DataFrame:\n{data.head()}")

            numeric_columns = data.select_dtypes(include=['number']).columns
            print(f"Numeric columns identified: {numeric_columns}")
            data[numeric_columns] = data[numeric_columns].fillna(0)

            data = data.dropna(how='all', axis=1)
            data = data.dropna(how='all', axis=0)

            print(f"Cleaned DataFrame with NaN filled:\n{data.head()}")

        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
            df = df.fillna(0)
            data = df.to_numpy()

        else:
            raise ValueError("Unsupported data type for BasicDataCleaningStep.")

        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("[BasicDataCleaningStep] All rows or columns were removed during cleaning.")

        context.set_processed_data(data)
        print("[BasicDataCleaningStep] Data cleaning completed.")


class AdvancedDataCleaningStep(PipelineStep):
    def process(self, context):
        data = context.get_processed_data()

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Unsupported data type for AdvancedDataCleaningStep. Only pandas DataFrame is supported.")

        print(f"Original DataFrame:\n{data.head()}")

        numeric_columns = data.select_dtypes(include=['number']).columns
        print(f"Numeric columns identified: {numeric_columns}")

        for col in numeric_columns:
            median_value = data[col].median()
            data[col].fillna(median_value, inplace=True)
            print(f"Filled NaN in column '{col}' with median value {median_value}.")

        threshold = 0.8
        data = data.dropna(axis=1, thresh=int(threshold * len(data)))
        data = data.dropna(axis=0, thresh=int(threshold * len(data.columns)))

        z_scores = np.abs((data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std())
        data = data[(z_scores < 3).all(axis=1)]

        print(f"Data after outlier removal:\n{data.head()}")

        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        initial_rows = data.shape[0]
        data = data.drop_duplicates()
        print(f"Removed {initial_rows - data.shape[0]} duplicate rows.")

        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("[AdvancedDataCleaningStep] All rows or columns were removed during cleaning.")

        context.set_processed_data(data)
        print("[AdvancedDataCleaningStep] Data cleaning completed.")


class NormalizationStep(PipelineStep):
    def process(self, context):
        print("[DataCleaningStep] Cleaning data...")
        data = context.get_processed_data()

        if isinstance(data, pd.DataFrame):
            print(f"Original DataFrame:\n{data.head()}")

            numeric_columns = data.select_dtypes(include=['number']).columns
            data[numeric_columns] = data[numeric_columns].fillna(0)

            categorical_columns = data.select_dtypes(include=['object', 'category']).columns
            print(f"Categorical columns identified: {categorical_columns}")

            for col in categorical_columns:
                unique_values = data[col].dropna().unique()
                if len(unique_values) > 0:
                    lookup_layer = ks.layers.StringLookup(vocabulary=unique_values, mask_token=None, num_oov_indices=1)
                    data[col] = lookup_layer(tf.constant(data[col].fillna(""))).numpy()

            data = data.dropna(how='all', axis=1)
            data = data.dropna(how='all', axis=0)

            print(f"Cleaned DataFrame:\n{data.head()}")

        elif isinstance(data, np.ndarray):

            df = pd.DataFrame(data)
            df = df.fillna(0)
            data = df.to_numpy()

        else:
            raise ValueError("Unsupported data type for DataCleaningStep.")

        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("[DataCleaningStep] All rows or columns were removed during cleaning.")

        context.set_processed_data(data)
        print("[DataCleaningStep] Data cleaning completed.")


class ClusteringStep(PipelineStep):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def process(self, context):
        print("[ClusteringStep] Performing clustering...")

        data = context.get_processed_data()

        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("[ClusteringStep] Data is empty or has no features. Clustering cannot proceed.")

        data = data.astype('float32')

        model = Sequential([
            Dense(128, activation='relu', input_shape=(data.shape[1],)),
            Dense(64, activation='relu'),
            Dense(self.n_clusters, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        random_indices = tf.convert_to_tensor(
            np.random.choice(self.n_clusters, size=data.shape[0]), dtype=tf.int32
        )
        dummy_labels = tf.gather(tf.eye(self.n_clusters), random_indices)

        model.fit(data, dummy_labels, epochs=5, verbose=1)

        clusters = model.predict(data)

        if isinstance(clusters, tf.Tensor):
            clusters = clusters.numpy()

        context.metadata['clusters'] = clusters
        context.set_processed_data(clusters)
        print("[ClusteringStep] Clustering completed.")


class MatrixCreationStep(PipelineStep):
    def process(self, context):
        print("[MatrixCreationStep] Creating similarity matrix...")
        data = context.get_processed_data()

        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        similarity_matrix = tf.linalg.matmul(data, data, transpose_b=True)
        context.set_processed_data(similarity_matrix.numpy())
        print("[MatrixCreationStep] Similarity matrix created.")


class FeatureEngineeringStep(PipelineStep):

    def __init__(self, feature_definitions):
        self.feature_definitions = feature_definitions

    def process(self, context):
        print("[FeatureEngineeringStep] Performing feature engineering...")

        data = context.get_processed_data()

        if not isinstance(data, pd.DataFrame):
            print("[FeatureEngineeringStep] Skipped feature engineering for non-DataFrame data.")
            return

        for new_feature, base_features in self.feature_definitions.items():
            if not all(feature in data.columns for feature in base_features):
                print(f"Warning: One or more base features for '{new_feature}' not found in data.")
                continue

            try:

                for feature in base_features:
                    data[feature] = pd.to_numeric(data[feature], errors='coerce')
                    print(f"Converted column '{feature}' to numeric.")

                data[new_feature] = data[base_features].prod(axis=1)
                print(f"[FeatureEngineeringStep] Created new feature '{new_feature}' based on {base_features}.")
            except Exception as e:
                print(f"Error creating feature '{new_feature}': {e}")

        context.set_processed_data(data)
        print("[FeatureEngineeringStep] Feature engineering completed.")



class DimensionalityReductionStep(PipelineStep):
    def __init__(self, reduced_dim):
        self.reduced_dim = reduced_dim

    def process(self, context):
        print("[DimensionalityReductionStep] Reducing dimensions...")
        data = context.get_processed_data()

        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("[DimensionalityReductionStep] Data is empty or has no features.")

        data = data.astype('float32')

        model = Sequential([
            Dense(self.reduced_dim, activation=None, input_shape=(data.shape[1],))
        ])

        reduced_data = model(data)

        if isinstance(reduced_data, tf.Tensor):
            reduced_data = reduced_data.numpy()

        context.set_processed_data(reduced_data)
        print(f"[DimensionalityReductionStep] Reduced dimensions to {self.reduced_dim}.")


class DataBalancingStep(PipelineStep):
    def process(self, context):
        print("[DataBalancingStep] Balancing data...")
        data = context.get_processed_data()

        if not isinstance(data, pd.DataFrame):
            raise ValueError("DataBalancingStep requires a DataFrame.")

        class_counts = data['label'].value_counts()
        print(f"Class distribution before balancing:\n{class_counts}")

        max_class = class_counts.idxmax()
        min_class = class_counts.idxmin()

        majority = data[data['label'] == max_class]
        minority = data[data['label'] == min_class]

        minority_oversampled = minority.sample(len(majority), replace=True)
        balanced_data = pd.concat([majority, minority_oversampled], axis=0)

        context.set_processed_data(balanced_data)
        print(f"[DataBalancingStep] Class distribution after balancing:\n{balanced_data['label'].value_counts()}")


class DataSavingStep(PipelineStep):
    def __init__(self, save_path):
        self.save_path = save_path

    def process(self, context):
        print("[DataSavingStep] Saving data...")
        data = context.get_processed_data()

        if isinstance(data, pd.DataFrame):
            data.to_csv(self.save_path, index=False)
            print(f"Data saved to {self.save_path}")
        else:
            raise ValueError("DataSavingStep requires a DataFrame.")

        print("[DataSavingStep] Data saving completed.")


class DataSplittingStep(PipelineStep):

    def __init__(self, split_type: str = "Holdout", test_size: float = 0.2, k: int = 5, stratified: bool = False, time_column: Optional[str] = None):
        self.split_type = split_type
        self.test_size = test_size
        self.k = k
        self.stratified = stratified
        self.time_column = time_column

    def process(self, context):
        data = context.get_processed_data()

        if not isinstance(data, pd.DataFrame):
            raise ValueError("[DataSplittingStep] Input data must be a pandas DataFrame.")

        if self.split_type == "Holdout":
            self._holdout_split(context, data)
        elif self.split_type == "Cross-Validation":
            self._cross_validation_split(context, data)
        elif self.split_type == "Time-Based Split":
            self._time_based_split(context, data)
        else:
            raise ValueError(f"[DataSplittingStep] Unsupported split type: {self.split_type}")

    def _holdout_split(self, context, data: pd.DataFrame):
        train_size = int(len(data) * (1 - self.test_size))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        context.metadata["train_data"] = train_data
        context.metadata["test_data"] = test_data
        print(f"[Holdout Split] Train size: {len(train_data)}, Test size: {len(test_data)}.")

    def _cross_validation_split(self, context, data: pd.DataFrame):
        kfold = ks.utils.split_dataset(
            dataset=data.values,
            num_splits=self.k,
            shuffle=True,
            seed=42
        )
        context.metadata["cv_splits"] = kfold
        print(f"[Cross-Validation] Created {self.k} folds.")

    def _time_based_split(self, context, data: pd.DataFrame):
        if self.time_column is None or self.time_column not in data.columns:
            raise ValueError(f"[Time-Based Split] Time column '{self.time_column}' must be provided and exist in the data.")

        data = data.sort_values(by=self.time_column)
        split_index = int(len(data) * (1 - self.test_size))
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]
        context.metadata["train_data"] = train_data
        context.metadata["test_data"] = test_data
        print(f"[Time-Based Split] Train size: {len(train_data)}, Test size: {len(test_data)}.")