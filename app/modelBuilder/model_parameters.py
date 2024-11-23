from typing import Any

class ModelParameters:
    def __init__(self, learning_rate: float = 0.01, regularization: float = 0.01, max_iterations: int = 1000, batch_size: int = 32, embedding_dim = 64):
        self.params = {
            'learning_rate': learning_rate,
            'regularization': regularization,
            'max_iterations': max_iterations,
            'batch_size': batch_size,
            'embedding_dim': embedding_dim
        }

    def set_parameter(self, name: str, value: Any):
        self.params[name] = value

    def get_parameter(self, name: str) -> Any:
        return self.params.get(name, None)