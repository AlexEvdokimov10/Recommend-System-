from typing import Dict, List
from .template import Template

class TemplateRegistry:
    def __init__(self):
        self._templates = {}

    def register_template(self, template: Template):
        self._templates[template.name] = template

    def get_template(self, name: str) -> Template:
        return self._templates.get(name)

    def get_all_templates(self) -> List[Template]:
        return list(self._templates.values())