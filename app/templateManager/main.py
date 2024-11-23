from flask import Flask, jsonify, request,Blueprint
from .attribute import Attribute
from .template_factory import TemplateFactory
from .template_registry import TemplateRegistry
from .storageHandler import FileStorageHandler, DatabaseStorageHandler
import os

tmp_bp= Blueprint('template_main', __name__) 

template_factory = TemplateFactory()
template_registry = TemplateRegistry()
file_storage = FileStorageHandler('templates')
db_storage = DatabaseStorageHandler()

@tmp_bp.route('/create_template', methods=['POST'])
def create_template():
    data = request.json
    name = data.get("name")
    attributes_data = data.get("attributes")


    if not name or not attributes_data:
        return jsonify({"error": "Template name and attributes are required"}), 400


    attributes = [
        Attribute(attr["name"], attr["description"], eval(attr["type"]), attr["modelType"])
        for attr in attributes_data
    ]


    template = template_factory.create_template(name, attributes)
    template_registry.register_template(template)
    

    if os.getenv('SAVE_TEMPATE_AS_FILE'):
        file_storage.save_template(template)
    if os.getenv('SAVE_TEMPALE_TO_DB'):  
        db_storage.save_template(template)

    return jsonify({"message": f"Template '{name}' created and saved successfully"}), 201

@tmp_bp.route('/get_template/<string:name>', methods=['GET'])
def get_template(name):

  
    template = template_registry.get_template(name)
    if not template:
        template = db_storage.load_template(name) or file_storage.load_template(name)
        if template:
            template_registry.register_template(template)
        else:
            return jsonify({"error": "Template not found"}), 404


    attributes = [{"name": attr.name, "description": attr.description, "type": attr.type.__name__, "modelType": attr.model_type} for attr in template.get_attributes()]
    return jsonify({"name": template.name, "attributes": attributes}), 200

@tmp_bp.route('/get_all_templates', methods=['GET'])
def get_all_templates():

    templates = template_registry.get_all_templates()
    templates_data = [
        {
            "name": template.name,
            "attributes": [
                {"name": attr.name, "description": attr.description, "type": attr.type.__name__, "modelType": attr.model_type}
                for attr in template.get_attributes()
            ]
        }
        for template in templates
    ]
    return jsonify(templates_data), 200

@tmp_bp.route('/delete_template/<string:name>', methods=['DELETE'])
def delete_template(name):
    template = template_registry.get_template(name)
    message = f"Template '{name}' deleted successfully from"
    if template:
        message += ' registry '
        template_registry._templates.pop(name)
   
    if(os.getenv('DELETE_TEMPATE_AS_FILE')):
        file_path = f"templates/{name}.json"
        if os.path.exists(file_path):
            os.remove(file_path)
            message += ' fileStroge '
    
    if(os.getenv('DELETE_TEMPALE_TO_DB')):
        db_storage.conn.execute("DELETE FROM templates WHERE name=?", (name,))
        db_storage.conn.commit()
        message += ' Database '

    return jsonify({"message": message}), 200