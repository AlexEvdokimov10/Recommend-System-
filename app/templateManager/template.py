from typing import List
from .attribute import Attribute

class Template:
    def __init__(self, name: str):
        self.name = name
        self.attributes = []

    def add_attribute(self, attribute: Attribute):
        self.attributes.append(attribute)

    def get_attributes(self) -> List[Attribute]:
        return self.attributes

class ModelTemplate:
    def __init__(self, model_name: str, parameters: dict):
        self.model_name = model_name
        self.parameters = parameters

    def get_model_name(self):

        return self.model_name

    def get_parameters(self):

        return self.parameters

    def update_parameters(self, new_params: dict):

        self.parameters.update(new_params)