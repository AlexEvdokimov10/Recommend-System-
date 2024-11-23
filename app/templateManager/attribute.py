class Attribute:
    def __init__(self, name: str, description: str, attr_type: type, model_type: str):
        self.name = name
        self.description = description
        self.type = attr_type
        self.model_type = model_type

    def __repr__(self):
        return f"Attribute(name={self.name}, type={self.type.__name__}, model_type={self.model_type})"

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type.__name__,  
            "model_type": self.model_type
        }