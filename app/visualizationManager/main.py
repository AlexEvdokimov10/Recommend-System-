import io
import os

from flask import Blueprint, jsonify, request, send_file
import pandas as pd
import matplotlib

from app.dataPipline.data_pipline import PipelineContext, BasicDataCleaningStep, AdvancedDataCleaningStep, DataPipeline, \
    DataBalancingStep, FeatureEngineeringStep
from app.repository.data_repository import DatabaseSource, DataMapper, DataRepository
from app.visualizationManager.visualizator import DataVisualizerFacade
from matplotlib import pyplot as plt

matplotlib.use('Agg')


visualization_bp = Blueprint("visualization", __name__)


data_source = DatabaseSource("db_source")
data_mapper = DataMapper()
data_repo = DataRepository(data_source, data_mapper)


visualizer = DataVisualizerFacade(output_path="visualizations")

def save_figure_to_buffer(format_type: str) -> io.BytesIO:
    buffer = io.BytesIO()
    plt.savefig(buffer, format=format_type, bbox_inches="tight")
    buffer.seek(0)
    return buffer


@visualization_bp.route("/visualize/evaluate", methods=["POST"])
def visualize_data_evaluation():
    try:
        table_name = request.json.get("table_name")
        format_type = request.json.get("format", "png")

        if not table_name:
            return jsonify({"error": "Table name is required"}), 400


        data = pd.DataFrame([row.__dict__ for row in data_repo.get_data(table_name)])
        if data.empty:
            return jsonify({"error": f"No data found in table '{table_name}'"}), 404

        visualizer.visualize_data_evaluation(data)

        buffer = save_figure_to_buffer(format_type)
        return send_file(buffer, mimetype=f"image/{format_type}", as_attachment=False)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@visualization_bp.route("/visualize/clean", methods=["POST"])
def visualize_clean_step():
    try:
        table_name = request.json.get("table_name")
        step_type = request.json.get("step_type", "basic")
        column_name = request.json.get("column_name", "anime_id")
        top_n = request.json.get("top_n", 10)
        format_type = request.json.get("format", "png")

        if not table_name:
            return jsonify({"error": "Table name is required"}), 400


        raw_data = pd.DataFrame([row.__dict__ for row in data_repo.get_data(table_name)])
        if raw_data.empty:
            return jsonify({"error": f"No data found in table '{table_name}'"}), 404


        context = PipelineContext(raw_data)
        pipeline = DataPipeline()

        if step_type == "basic":
            pipeline.add_step(BasicDataCleaningStep())
        elif step_type == "advanced":
            pipeline.add_step(AdvancedDataCleaningStep())
        else:
            return jsonify({"error": f"Unsupported step_type '{step_type}'"}), 400

        pipeline.execute(context)
        cleaned_data = context.get_processed_data()


        visualizer.visualize_clean(raw_data, pd.DataFrame(cleaned_data), column_name, top_n)

        buffer = save_figure_to_buffer(format_type)
        return send_file(buffer, mimetype=f"image/{format_type}", as_attachment=False)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@visualization_bp.route("/visualize/class_balance", methods=["POST"])
def visualize_class_balance():
    try:
        table_name = request.json.get("table_name")
        column_name = request.json.get("column_name")
        format_type = request.json.get("format", "png")
        top_n = request.json.get("top_n", 10)

        if not table_name or not column_name:
            return jsonify({"error": "Both table_name and column_name are required"}), 400


        raw_data = pd.DataFrame([row.__dict__ for row in data_repo.get_data(table_name)])
        if raw_data.empty:
            return jsonify({"error": f"No data found in table '{table_name}'"}), 404
        if column_name not in raw_data.columns:
            return jsonify({"error": f"Column '{column_name}' not present in table"}), 404

        context = PipelineContext(raw_data)
        pipeline = DataPipeline()

        pipeline.add_step(DataBalancingStep())

        pipeline.execute(context)
        balanced_data = context.get_processed_data()

        if column_name not in balanced_data.columns:
            return jsonify({"error": f"Column '{column_name}' not found in balanced data"}), 404

        class_counts = balanced_data[column_name].value_counts().head(top_n)
        if class_counts.empty:
            return jsonify({"error": f"No data to visualize for column '{column_name}'"}), 404

        visualizer.visualize_class_balance(
            pd.DataFrame({column_name: class_counts.index, 'count': class_counts.values}),
            column_name
        )

        buffer = save_figure_to_buffer(format_type)
        return send_file(buffer, mimetype=f"image/{format_type}", as_attachment=False)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@visualization_bp.route("/visualize/outliers", methods=["POST"])
def visualize_outliers():
    try:
        table_name = request.json.get("table_name")
        column_name = request.json.get("column_name")
        format_type = request.json.get("format", "png")

        if not table_name or not column_name:
            return jsonify({"error": "Both table_name and column_name are required"}), 400


        data = pd.DataFrame([row.__dict__ for row in data_repo.get_data(table_name)])
        if data.empty:
            return jsonify({"error": f"No data found in table '{table_name}'"}), 404

        visualizer.visualize_outliers(data, column_name)


        buffer = save_figure_to_buffer(format_type)
        return send_file(buffer, mimetype=f"image/{format_type}", as_attachment=False)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400


@visualization_bp.route("/visualize/feature_engineering", methods=["POST"])
def visualize_feature_engineering():
    try:

        table_name = request.json.get("table_name")
        feature_definitions = request.json.get("feature_definitions")
        format_type = request.json.get("format", "png")


        if not table_name:
            return jsonify({"error": "Table name is required"}), 400
        if not feature_definitions:
            return jsonify({"error": "Feature definitions are required"}), 400


        raw_data = pd.DataFrame([row.__dict__ for row in data_repo.get_data(table_name)])
        if raw_data.empty:
            return jsonify({"error": f"No data found in table '{table_name}'"}), 404


        context = PipelineContext(raw_data)
        pipeline = DataPipeline()


        feature_step = FeatureEngineeringStep(feature_definitions)
        pipeline.add_step(feature_step)


        pipeline.execute(context)
        engineered_data = context.get_processed_data()


        output_path = "visualizations"
        os.makedirs(output_path, exist_ok=True)
        image_path = f"{output_path}/{table_name}_feature_relationships.{format_type}"
        visualizer.visualize_feature_relationships(engineered_data, feature_definitions)
        plt.savefig(image_path, format=format_type)
        plt.close()


        return send_file(
            image_path,
            mimetype=f"image/{format_type}",
            as_attachment=False
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500