from typing import List
from .template import Template,ModelTemplate
from .attribute import Attribute

class TemplateFactory:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super(TemplateFactory, cls).__new__(cls)
        return cls.instance

    def create_template(self, name: str, attributes: List[Attribute]) -> Template:
        template = Template(name)
        for attribute in attributes:
            template.add_attribute(attribute)
        return template
    
    @staticmethod
    def create_template_model(template_type: str, parameters:dict) -> ModelTemplate:
        if template_type == "neural_network":
            if not parameters:
                    parameters={
                        "learning_rate": 0.01,
                        "max_iterations": 100,
                        "epochs": 5 ,
                        "batch_size": 32,
                    }

            return ModelTemplate (
                model_name="Neural Collaborative Filtering",
                parameters=parameters
            )
        
        elif template_type == "collaborative_filtering":

            if not parameters:
                    parameters={
                        "regularization": 0.1,
                        "max_iterations": 50,
                    }
            return ModelTemplate(
                model_name="Collaborative Filtering",
                parameters=parameters
            )
        
        elif template_type == "content_based":
            
            if not parameters:
                parameters={"similarity_metric": "cosine",}

            return ModelTemplate(
                model_name="Content Based Filtering",
                parameters=parameters
            )
        else:
            raise ValueError(f"Unknown template type: {template_type}")