"""
Schema 校验器

验证生成的 A2UI JSON 符合规范
"""

import json
from pathlib import Path
from typing import Any

import jsonschema


class SchemaValidator:
    """A2UI Schema 校验器"""

    def __init__(self, schema_path: str | Path | None = None):
        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "schemas" / "a2ui_components.json"

        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

        self.component_types = list(self.schema.get("definitions", {}).keys())

    def validate_component(self, component: dict[str, Any]) -> tuple[bool, str | None]:
        """验证单个组件"""
        component_type = component.get("type")

        if not component_type:
            return False, "Missing 'type' field"

        if not component.get("id"):
            return False, "Missing 'id' field"

        if not component.get("props"):
            return False, "Missing 'props' field"

        # 基本结构验证
        required_props = self._get_required_props(component_type)
        props = component.get("props", {})

        for prop in required_props:
            if prop not in props:
                return False, f"Missing required prop: {prop}"

        return True, None

    def _get_required_props(self, component_type: str) -> list[str]:
        """获取组件必需的 props"""
        required_map = {
            "map_card": ["title"],
            "ar_label": ["text"],
            "comparison_card": ["title", "items"],
            "direction_arrow": ["direction"],
            "nutrition_card": ["product_name"],
            "price_calculator": ["items"],
            "task_card": ["title"],
            "step_card": ["title", "steps"],
        }
        return required_map.get(component_type, [])

    def validate_batch(
        self, components: list[dict[str, Any]]
    ) -> tuple[list[dict], list[tuple[int, str]]]:
        """批量验证组件"""
        valid = []
        errors = []

        for i, component in enumerate(components):
            is_valid, error = self.validate_component(component)
            if is_valid:
                valid.append(component)
            else:
                errors.append((i, error))

        return valid, errors
