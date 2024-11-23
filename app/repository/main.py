from flask import Flask, jsonify, request, Blueprint, Response
import xml.etree.ElementTree as ET
import pandas as pd

from .data_repository import DatabaseSource, DataMapper, DataRepository

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

rest_bp = Blueprint('repository_main', __name__)

db_source = DatabaseSource("db_source")
data_mapper = DataMapper()
data_repo = DataRepository(db_source, data_mapper)


def format_response(data, format_type="json"):
    if format_type == "json":
        print(data)
        return jsonify(data), 200
    elif format_type == "xml":
        root = ET.Element("response")
        for item in data:
            child = ET.SubElement(root, "item")
            for key, value in item.items():
                subchild = ET.SubElement(child, key)
                subchild.text = str(value)
        return Response(ET.tostring(root), mimetype="application/xml"), 200
    elif format_type == "text":
        return Response("\n".join(f"{k}: {v}" for d in data for k, v in d.items()), mimetype="text/plain"), 200
    elif format_type == "html":
        html_content = "<html><body>" + "".join(
            f"<p><b>{k}:</b> {v}</p>" for d in data for k, v in d.items()
        ) + "</body></html>"
        return Response(html_content, mimetype="text/html"), 200
    return jsonify({"error": "Unsupported format"}), 400


@rest_bp.route('/load_csv', methods=['POST'])
def load_csv():
    csv_file_path = request.json.get('csv_file_path')
    table_name = request.json.get('table_name')

    if not csv_file_path or not table_name:
        return jsonify({"error": "csv_file_path and table_name are required"}), 400

    try:
        db_source.load_csv_to_db(csv_file_path, table_name)
        return jsonify({"message": f"Data from {csv_file_path} loaded into table '{table_name}'"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@rest_bp.route('/data', methods=['GET'])
def get_data():
    try:
        table_name = request.args.get("table_name", "anime_test_table")
        format_type = request.headers.get("Accept", "application/json").split("/")[-1]

        data = data_repo.get_data(table_name)

        if data:
            formatted_data = [d.__dict__ for d in data]
            return format_response(formatted_data, format_type)
        else:
            return jsonify({"error": "No data found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@rest_bp.route('/load_data_file', methods=['POST'])
def load_data_file():
    file = request.files.get('file')
    table_name = request.form.get('table_name')

    if not file or not table_name:
        return jsonify({"error": "file and table_name are required"}), 400

    filename = file.filename
    file_extension = filename.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            df = pd.read_csv(file.stream)
        elif file_extension == 'json':
            df = pd.read_json(file.stream)
        elif file_extension == 'xml':
            tree = ET.parse(file.stream)
            root = tree.getroot()
            data = [{child.tag: child.text for child in item} for item in root]
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": f"Unsupported file type: {file_extension}"}), 400

        data_repo.save_dataframe(df, table_name)

        return jsonify({"message": f"Data from {filename} loaded into table '{table_name}'"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@rest_bp.errorhandler(Exception)
def handle_exception(e):
    response = {
        "error": "Server Error",
        "message": str(e),
    }

    if hasattr(e, "code"):
        return jsonify(response), e.code

    return jsonify(response), 500


@rest_bp.errorhandler(404)
def handle_404(e):
    response = {
        "error": "Not Found",
        "message": "The requested resource was not found on the server.",
    }
    return jsonify(response), 404


@rest_bp.errorhandler(400)
def handle_400(e):
    response = {
        "error": "Bad Request",
        "message": "Invalid request. Please check your input and try again.",
    }
    return jsonify(response), 400




