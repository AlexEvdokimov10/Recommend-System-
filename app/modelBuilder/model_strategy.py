from typing import List, Any, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tf_keras import layers, Model, optimizers, Sequential, metrics, callbacks
from tf_keras.src import regularizers

from .data_preprocesor import DataPreprocessor
from .evulation_metric import EvaluationMetrics
from .model_parameters import ModelParameters
from .models import NoBaseClassMovielensModel
from .training_data import TrainingData
from .utils import CustomRetrievalTask, CustomCallback


class ModelStrategy:
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        clean_data = DataPreprocessor.clean(data)
        normalized_data = DataPreprocessor.normalize(clean_data)
        return normalized_data

    def build(self, data: TrainingData, parameters: ModelParameters, key_attributes: Dict[str, str]):
        raise NotImplementedError


    def predict(self, input_data: pd.DataFrame, key_attributes: Dict[str, str]) -> List[Any]:
        raise NotImplementedError


    def evaluate(self, data: pd.DataFrame, labels: pd.DataFrame, key_attributes: Dict[str, str]) -> EvaluationMetrics:
        predictions = self.predict(data, key_attributes)
        actuals = labels.values.flatten().tolist()
        return EvaluationMetrics.calculate_metrics(predictions, actuals)


class CollaborativeFiltering(ModelStrategy):

    def __init__(self):
        self.neural_network = None

    def build(self, data: TrainingData, parameters: ModelParameters, key_attributes: Dict[str, str]):

        processed_data = self.preprocess_data(data.features)
        user_key = key_attributes.get("user_key", "user_id")
        item_key = key_attributes.get("item_key", "item_id")

        if user_key not in processed_data.columns or item_key not in processed_data.columns:
            raise ValueError(f"Missing keys in data: {user_key}, {item_key}")

        max_user_id = processed_data[user_key].max()
        max_item_id = processed_data[item_key].max()
        num_users = max_user_id + 1
        num_items = max_item_id + 1

        user_input = layers.Input(shape=(1,), name="user_input")
        item_input = layers.Input(shape=(1,), name="item_input")

        user_embedding = layers.Embedding(
            input_dim=num_users,
            output_dim=parameters.get_parameter("embedding_dim"),
            name="user_embedding"
        )(user_input)
        item_embedding = layers.Embedding(
            input_dim=num_items,
            output_dim=parameters.get_parameter("embedding_dim"),
            name="item_embedding"
        )(item_input)

        concat = layers.Concatenate()([
            layers.Flatten()(user_embedding),
            layers.Flatten()(item_embedding)
        ])
        dense_1 = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(concat)
        dropout_1 = layers.Dropout(0.3)(dense_1)
        dense_2 = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(dropout_1)
        dropout_2 = layers.Dropout(0.3)(dense_2)
        dense_3 = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(dropout_2)

        is_multiclass = len(data.labels.shape) > 1 and data.labels.shape[1] > 1
        if is_multiclass:
            num_classes = data.labels.shape[1]
            output = layers.Dense(num_classes, activation="softmax", name="output")(dense_3)
            loss_function = "categorical_crossentropy"
        else:
            output = layers.Dense(1, activation="sigmoid", name="output")(dense_3)
            loss_function = "binary_crossentropy"

        self.neural_network = Model(inputs=[user_input, item_input], outputs=output)
        self.neural_network.compile(
            optimizer=optimizers.Adam(learning_rate=parameters.get_parameter("learning_rate")),
            loss=loss_function,
            metrics=["accuracy", metrics.Precision(), metrics.Recall()]
        )


        x = np.array([processed_data[user_key].values, processed_data[item_key].values]).T
        y = data.labels.values

        if not is_multiclass:
            rare_class_indices = np.where(y == 1)[0]
            oversample_count = len(y) - len(rare_class_indices)
            oversampled_indices = np.random.choice(rare_class_indices, oversample_count, replace=True)
            balanced_indices = np.concatenate([np.arange(len(y)), oversampled_indices])
            x_balanced = x[balanced_indices]
            y_balanced = y[balanced_indices]
        else:
            x_balanced, y_balanced = x, y

        self.neural_network.fit(
            x=[x_balanced[:, 0], x_balanced[:, 1]],
            y=y_balanced,
            batch_size=parameters.get_parameter("batch_size"),
            epochs=parameters.get_parameter("max_iterations"),
            validation_split=0.2,
            callbacks=[CustomCallback()],
            verbose=1
        )

    def evaluate(self, data: pd.DataFrame, labels: pd.DataFrame, key_attributes: Dict[str, str]) -> EvaluationMetrics:

        processed_data = self.preprocess_data(data)

        user_key = key_attributes.get("user_key", "user_id")
        item_key = key_attributes.get("item_key", "item_id")

        num_users = self.neural_network.get_layer("user_embedding").input_dim
        num_items = self.neural_network.get_layer("item_embedding").input_dim

        valid_user_mask = processed_data[user_key] < num_users
        valid_item_mask = processed_data[item_key] < num_items

        filtered_data = processed_data[valid_user_mask & valid_item_mask]
        filtered_labels = labels[valid_user_mask & valid_item_mask]

        if filtered_data.empty:
            raise ValueError("No valid data for evaluation after filtering invalid user or item IDs.")


        predictions = self.predict(filtered_data, key_attributes)

        metrics = EvaluationMetrics.calculate_metrics(predictions, filtered_labels.values.flatten().tolist())

        return metrics

    def predict(self, input_data: pd.DataFrame, key_attributes: Dict[str, str]) -> List[float]:


        processed_data = self.preprocess_data(input_data)


        user_key = key_attributes.get("user_key", "user_id")
        item_key = key_attributes.get("item_key", "item_id")

        missing_keys = [key for key in [user_key, item_key] if key not in processed_data.columns]
        if missing_keys:
            raise ValueError(f"Missing keys in input data: {missing_keys}")

        num_users = self.neural_network.get_layer("user_embedding").input_dim
        num_items = self.neural_network.get_layer("item_embedding").input_dim

        processed_data[user_key] = processed_data[user_key].apply(lambda x: x if x < num_users else 0)
        processed_data[item_key] = processed_data[item_key].apply(lambda x: x if x < num_items else 0)
        predictions = self.neural_network.predict(
            [processed_data[user_key].values, processed_data[item_key].values]
        )

        return predictions.flatten().tolist()


class ModelBasedFiltering(CollaborativeFiltering):
    pass


class NeuralNetworkCollaborativeFiltering(ModelBasedFiltering):

    def build(self, data: TrainingData, parameters: ModelParameters, key_attributes: Dict[str, str]):
        processed_data = self.preprocess_data(data.features)

        user_key = key_attributes.get("user_key", "user_id")
        item_key = key_attributes.get("item_key", "item_id")

        num_users = processed_data[user_key].nunique()
        num_items = processed_data[item_key].nunique()
        embedding_dim = parameters.get_parameter("embedding_dim")

        user_input = layers.Input(shape=(1,), name="user_input")
        item_input = layers.Input(shape=(1,), name="item_input")

        user_embedding = layers.Embedding(num_users + 1, embedding_dim, name="user_embedding")(user_input)
        item_embedding = layers.Embedding(num_items + 1, embedding_dim, name="item_embedding")(item_input)

        user_vec = layers.Flatten()(user_embedding)
        item_vec = layers.Flatten()(item_embedding)

        concat = layers.Concatenate()([user_vec, item_vec])
        dense_1 = layers.Dense(128, activation="relu")(concat)
        dense_2 = layers.Dense(64, activation="relu")(dense_1)
        dense_3 = layers.Dense(32, activation="relu")(dense_2)
        output = layers.Dense(1, activation="sigmoid")(dense_3)

        self.neural_network = Model(inputs=[user_input, item_input], outputs=output)
        self.neural_network.compile(
            optimizer=optimizers.Adam(learning_rate=parameters.get_parameter("learning_rate")),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        self.neural_network.fit(
            x=[processed_data[user_key].values, processed_data[item_key].values],
            y=data.labels.values,
            batch_size=parameters.get_parameter("batch_size"),
            epochs=parameters.get_parameter("max_iterations"),
            verbose=1
        )


class MovieRecommendationStrategy(ModelBasedFiltering):
    def __init__(self):
        self.user_model = None
        self.item_model = None
        self.task = None
        self.trained_model = None

    def preprocess_data(self, data: Any) -> Any:
        return data

    def build(self, data: TrainingData, parameters: ModelParameters, key_attributes: Dict[str, str]):
        processed_data = self.preprocess_data(data.features)
        embedding_dim = parameters.get_parameter("embedding_dim")


        user_key = key_attributes.get("user_key", "user_id")
        item_key = key_attributes.get("item_key", "item_id")


        unique_users = processed_data[user_key].astype(str).unique()
        unique_items = processed_data[item_key].astype(str).unique()


        self.user_model = Sequential([
            layers.StringLookup(vocabulary=unique_users, mask_token=None),
            layers.Embedding(len(unique_users) + 1, embedding_dim)
        ])


        self.item_model = Sequential([
            layers.StringLookup(vocabulary=unique_items, mask_token=None),
            layers.Embedding(len(unique_items) + 1, embedding_dim)
        ])


        self.task = CustomRetrievalTask(custom_metrics=[metrics.Precision(), metrics.Recall()])

    def train(self, data: TrainingData, parameters: ModelParameters, key_attributes: Dict[str, str]):
        processed_data = self.preprocess_data(data.features)

        user_key = key_attributes.get("user_key", "user_id")
        item_key = key_attributes.get("item_key", "item_id")

        user_ids = processed_data[user_key].astype(str)
        item_ids = processed_data[item_key].astype(str)
        labels = data.labels.values.flatten()

        self.trained_model = NoBaseClassMovielensModel(self.user_model, self.item_model, self.task)
        self.trained_model.compile(
            optimizer=optimizers.Adam(learning_rate=parameters.get_parameter("learning_rate"))
        )

        dataset = tf.data.Dataset.from_tensor_slices({
            "user_id": user_ids,
            "item_id": item_ids,
            "label": labels
        }).batch(parameters.get_parameter("batch_size"))

        self.trained_model.fit(dataset, epochs=parameters.get_parameter("max_iterations"), verbose=1)

    def evaluate(self, data: Any, labels: Any, key_attributes: Dict[str, str]) -> Dict[str, Any]:
        processed_data = self.preprocess_data(data)

        user_key = key_attributes.get("user_key", "user_id")
        item_key = key_attributes.get("item_key", "item_id")

        user_ids = processed_data[user_key].astype(str)
        item_ids = processed_data[item_key].astype(str)

        user_embeddings = self.user_model(user_ids)
        item_embeddings = self.item_model(item_ids)

        loss = self.task(user_embeddings, item_embeddings, labels.values.flatten())
        metrics = self.task.compute_metrics()

        return {"loss": loss.numpy(), "metrics": metrics}

    def predict(self, data: Any, key_attributes: Dict[str, str]) -> np.ndarray:
        processed_data = self.preprocess_data(data)

        user_key = key_attributes.get("user_key", "user_id")
        item_key = key_attributes.get("item_key", "item_id")

        user_ids = processed_data[user_key].astype(str)
        item_ids = processed_data[item_key].astype(str)

        predictions = self.trained_model({
            "user_id": user_ids,
            "item_id": item_ids
        })

        return predictions.numpy()


class ContentBasedFiltering(ModelStrategy):

    def build(self, data: TrainingData, parameters: ModelParameters, key_attributes: Dict[str, str]):
        return [0]

    def predict(self, input_data: pd.DataFrame, key_attributes: Dict[str, str]) -> List[Any]:
        return [0] * len(input_data)
