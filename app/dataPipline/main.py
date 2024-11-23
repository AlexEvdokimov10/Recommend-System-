from flask import Blueprint, jsonify, request
from app.repository.data_repository import DataRepository, DatabaseSource, DataMapper


from .utils import default_process_data, save_processed_data

data_pipeline_bp = Blueprint("data_pipeline", __name__)


data_source = DatabaseSource("db_source")
data_mapper = DataMapper()
data_repo = DataRepository(data_source, data_mapper)

default_steps = [
    'data_cleaning_step', 'data_evaluation_step', 'normalization_step', 'categorical_encoding_step',
    'data_balancing_step', 'splitting_step', 'feature_engineering_step', 'dimensionality_reduction_step'
]

is_save_processed_data = True

@data_pipeline_bp.route("/process_data", methods=["POST"])
def process_data():
    try:
        table_name = request.json.get("table_name")
        steps = request.json.get("steps") if request.json.get("steps") else default_steps
        save_table_name = request.json.get("save_table_name", 'processed_data_anime_frame')

        if not table_name:
            return jsonify({"error": "Table name is required"}), 400


        raw_data = data_repo.get_data(table_name)
        if not raw_data:
            return jsonify({"error": f"No data found in table '{table_name}'"}), 404

        processed_data = default_process_data(raw_data, steps)


        if is_save_processed_data:
            save_processed_data(processed_data, save_table_name)

        return jsonify({"processed_data": processed_data.tolist()}), 200

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@data_pipeline_bp.route("/list_steps", methods=["GET"])
def list_steps():

    available_steps = {
        "data_cleaning_step": "Cleaning Data",
        "data_evaluation_step": "Data evaluation",
        "normalization_step": "Data Normalization",
        "categorical_encoding_step": "Encoding categorical data",
        "data_balancing_step": "Class Balancing",
        "splitting_step": "Data separation",
        "feature_engineering_step": "Feature generation",
        "dimensionality_reduction_step": "Dimensionality reduction"
    }
    return jsonify({"available_steps": available_steps}), 200


@data_pipeline_bp.route("/save_processed_data", methods=["POST"])
def save_processed_data_endpoint():

    try:
        processed_data = request.json.get("processed_data")
        save_table_name = request.json.get("save_table_name", 'processed_data_anime_frame')

        if not processed_data:
            return jsonify({"error": "Processed data is required"}), 400

        save_processed_data(processed_data, save_table_name)
        return jsonify({"message": f"Processed data saved to table '{save_table_name}'"}), 200

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500


