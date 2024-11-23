import json
import os
import psycopg2
from abc import ABC, abstractmethod
from .template import Template
from .attribute import Attribute
from ..utils import get_database_url

DATABASE_URL = get_database_url('dev')


class Attribute:

    def __init__(self, attr_name, attr_value):
        self.attr_name = attr_name
        self.attr_value = attr_value

    def to_dict(self):
        return {
            "attr_name": self.attr_name,
            "attr_value": self.attr_value
        }

class StorageHandler(ABC):
    @abstractmethod
    def save_template(self, template: Template):
        pass

    @abstractmethod
    def load_template(self, name: str) -> Template:
        pass

class FileStorageHandler(StorageHandler):
    def __init__(self, directory: str):
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def save_template(self, template: Template):
        path = os.path.join(self.directory, f"{template.name}.json")
        with open(path, 'w') as file:
            json.dump({
                "name": template.name,
                "attributes": [attr.to_dict() for attr in template.get_attributes()]
            }, file)

    def load_template(self, name: str) -> Template:
        path = os.path.join(self.directory, f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path, 'r') as file:
            data = json.load(file)
            template = Template(data["name"])
            for attr in data["attributes"]:
                template.add_attribute(Attribute(**attr))
            return template

class DatabaseStorageHandler(StorageHandler):
    def __init__(self):
        self.conn = psycopg2.connect(DATABASE_URL)
        self._create_table()

    def _create_table(self):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS templates (
                    name TEXT PRIMARY KEY,
                    attributes JSONB
                )
            """)
            self.conn.commit()

    def save_template(self, template: Template):
        attributes_json = json.dumps([attr.to_dict() for attr in template.get_attributes()])
        with self.conn, self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO templates (name, attributes)
                VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE
                SET attributes = EXCLUDED.attributes
            """, (template.name, attributes_json))

    def load_template(self, name: str) -> Template:
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT attributes FROM templates WHERE name = %s", (name,))
            row = cursor.fetchone()
            if row:
                attributes = json.loads(row[0])
                template = Template(name)
                for attr in attributes:
                    template.add_attribute(Attribute(**attr))
                return template
        return None

    def close_connection(self):
        if self.conn:
            self.conn.close()

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Attribute):
            return obj.to_dict()
        return super().default(obj)