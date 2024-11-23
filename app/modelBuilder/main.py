import logging
import os
import time
from datetime import datetime

from flask import Blueprint, jsonify, request
from .data_preprocesor import DataPreprocessor
from .modelBuilder import ModelBuilder, ModelEvaluator
from .model_strategy import (
    NeuralNetworkCollaborativeFiltering,
    ContentBasedFiltering,
    ModelBasedFiltering,
    CollaborativeFiltering,
    MovieRecommendationStrategy,
)
from .model_parameters import ModelParameters
from .training_data import TrainingData
import tf_keras as ks
import tensorflow as tf
import pandas as pd
from ..repository.data_repository import DatabaseSource, DataRepository, DataMapper
from ..utils import to_dataframe


model_builder_bp = Blueprint("model_builder", __name__)


data_source = DatabaseSource("db_source")
data_mapper = DataMapper()
data_repo = DataRepository(data_source, data_mapper)


MODELS_DIR = "models"
STRATEGIES  = {
    "CollaborativeFiltering": CollaborativeFiltering,
    "ContentBasedFiltering": ContentBasedFiltering,
    "NeuralNetworkCollaborativeFiltering": NeuralNetworkCollaborativeFiltering,
    "ModelBasedFiltering": ModelBasedFiltering,
    "MovieRecommendationStrategy": MovieRecommendationStrategy,
}


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

@model_builder_bp.route('/train_model', methods=['POST'])
def train_model():
    try:
        logger.info("Getting request parameters")
        table_name = request.json.get("table_name")
        ratings_table = request.json.get("ratings_table")
        strategy_name = request.json.get("strategy")
        parameters = request.json.get("parameters", {})
        keyAttributes = request.json.get("keyAttributes", {})

        if not keyAttributes:
            logger.error("Key attributes are missing")
            return jsonify({"error": "Key attributes must be provided"}), 400

        merge_keys = list(keyAttributes.values())

        if not table_name or not ratings_table or not strategy_name:
            logger.error("Mandatory parameters are missing")
            return jsonify({"error": "Table name, ratings table, and strategy name are required"}), 400

        logger.info("Loading data from tables")
        raw_data = data_repo.get_data(table_name)
        if not raw_data:
            logger.error(f"Data not found in table '{table_name}'")
            return jsonify({"error": f"No data found in table '{table_name}'"}), 404

        raw_ratings = data_repo.get_data(ratings_table)
        if not raw_ratings:
            logger.error(f"Data not found in the league table '{ratings_table}'")
            return jsonify({"error": f"No data found in ratings table '{ratings_table}'"}), 404

        logger.info("Converting data to DataFrame")
        data = to_dataframe(raw_data)
        ratings = to_dataframe(raw_ratings)

        common_keys = [key for key in merge_keys if key in data.columns and key in ratings.columns]

        if not common_keys:
            logger.error("No common keys found between item table and ratings table")
            return jsonify({"error": "No common keys found between item table and ratings table"}), 400

        logger.info("Data Merging")
        merged_data = data.merge(ratings, on=common_keys, how='inner')
        logger.info(f"Data successfully merged on keys: {common_keys}")

        logger.info("Data preprocessing")
        clean_data = DataPreprocessor.clean(merged_data)
        normalized_data = DataPreprocessor.normalize(clean_data)

        logger.info("Splitting data into training and testing")
        tf.random.set_seed(42)
        dataset = tf.data.Dataset.from_tensor_slices(
            (normalized_data[merge_keys].values, normalized_data.drop(columns=merge_keys).values)
        )
        shuffled = dataset.shuffle(buffer_size=100_000, seed=42, reshuffle_each_iteration=False)
        train_dataset = shuffled.take(80_000)
        test_dataset = shuffled.skip(80_000).take(20_000)

        logger.info("Preparing data for the model")
        train_features = list(train_dataset.map(lambda x, y: x).as_numpy_iterator())
        train_labels = list(train_dataset.map(lambda x, y: y).as_numpy_iterator())
        test_features = list(test_dataset.map(lambda x, y: x).as_numpy_iterator())
        test_labels = list(test_dataset.map(lambda x, y: y).as_numpy_iterator())

        training_data = TrainingData(pd.DataFrame(train_features, columns=merge_keys), pd.DataFrame(train_labels))
        testing_data = TrainingData(pd.DataFrame(test_features, columns=merge_keys), pd.DataFrame(test_labels))

        logger.info("Setting up model parameters")
        model_params = ModelParameters(
            learning_rate=parameters.get("learning_rate"),
            max_iterations=parameters.get("max_iterations"),
            batch_size=parameters.get("batch_size"),
        )

        logger.info("Select strategy")
        strategy_class = STRATEGIES.get(strategy_name)
        if not strategy_class:
            logger.error(f"Unknown strategy: {strategy_name}")
            return jsonify({"error": f"Unknown strategy: {strategy_name}"}), 400

        logger.info(f"Start strategy: {strategy_name}")
        strategy = strategy_class()
        model_builder = ModelBuilder(model_params)
        model_builder.set_strategy(strategy)

        start_time = time.time()
        model_builder.build_model(training_data, keyAttributes)
        end_time = time.time()

        training_time = end_time - start_time

        logger.info("Save model")
        model_name = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_dir = os.path.join(MODELS_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model")
        ks.models.save_model(strategy.neural_network, model_path)

        logger.info("Model evaluation")
        evaluator = ModelEvaluator(strategy)
        metrics = evaluator.evaluate(testing_data.features, testing_data.labels, keyAttributes)

        logger.info("Create a report")
        report_path = os.path.join(model_dir, "report.txt")
        with open(report_path, "w") as report_file:
            report_file.write(f"Model Name: {model_name}\n")
            report_file.write(f"Saved At: {datetime.now().isoformat()}\n")
            report_file.write(f"Strategy: {strategy_name}\n")
            report_file.write(f"Parameters: {parameters}\n")
            report_file.write(f"Key Attributes: {keyAttributes}\n")
            report_file.write(f"Training Time: {training_time:.2f} seconds\n")
            report_file.write(f"Evaluation Metrics:\n")
            report_file.write(f"  Accuracy: {metrics.accuracy}\n")
            report_file.write(f"  Precision: {metrics.precision}\n")
            report_file.write(f"  Recall: {metrics.recall}\n")

        logger.info("Model trained and saved successfully")
        return jsonify({
            "message": "Model trained and saved successfully",
            "model_path": model_path,
            "evaluation_metrics": {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
            }
        }), 200
    except Exception as e:
        logger.exception("Error during model training")
        return jsonify({"error": str(e)}), 500


# @model_builder_bp.route('/evaluate_model', methods=['POST'])
# def evaluate_model():
#     try:
#
#         table_name = request.json.get("table_name")
#         strategy_name = request.json.get("strategy")
#         model_name = request.json.get("model_name")
#
#         if not table_name or not strategy_name or not model_name:
#             return jsonify({"error": "Table name, strategy name, and model name are required"}), 400
#
#
#         raw_data = data_repo.get_data(table_name)
#         if not raw_data:
#             return jsonify({"error": f"No data found in table '{table_name}'"}), 404
#
#
#         data = to_dataframe(raw_data)
#         clean_data = DataPreprocessor.clean(data)
#         normalized_data = DataPreprocessor.normalize(clean_data)
#
#         test_features = normalized_data.drop(columns=['label'])
#         test_labels = normalized_data['label']
#
#
#         model_path = os.path.join(MODELS_DIR, model_name, "model")
#         if not os.path.exists(model_path):
#             return jsonify({"error": f"Model '{model_name}' not found"}), 404
#
#         model = ks.models.load_model(model_path)
#
#         strategies = {
#             "CollaborativeFiltering": CollaborativeFiltering,
#             "ContentBasedFiltering": ContentBasedFiltering,
#             "NeuralNetworkCollaborativeFiltering": NeuralNetworkCollaborativeFiltering,
#             "ModelBasedFiltering": ModelBasedFiltering,
#             "MovieRecommendationStrategy": MovieRecommendationStrategy,
#         }
#         strategy_class = strategies.get(strategy_name)
#         if not strategy_class:
#             return jsonify({"error": f"Unknown strategy: {strategy_name}"}), 400
#
#         strategy = strategy_class()
#         strategy.neural_network = model
#
#         evaluator = ModelEvaluator(strategy)
#         metrics = evaluator.evaluate(test_features, test_labels,)
#
#         return jsonify({
#             "accuracy": metrics.accuracy,
#             "precision": metrics.precision,
#             "recall": metrics.recall,
#         }), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
