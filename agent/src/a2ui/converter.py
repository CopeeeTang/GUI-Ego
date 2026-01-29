"""A2UI format converter for transforming business components to A2UI v0.9 standard."""

import logging
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Mapping from business component types to A2UI atomic component structures
COMPONENT_MAPPING = {
    "ar_label": {
        "structure": "Card -> Row -> [Icon, Column -> [Text(h3), Text(caption)]]",
        "builder": "_build_ar_label",
    },
    "map_card": {
        "structure": "Card -> Column -> [Image, Text, Button]",
        "builder": "_build_map_card",
    },
    "direction_arrow": {
        "structure": "Card -> Row -> [Icon(arrowForward), Column -> [Text, Text]]",
        "builder": "_build_direction_arrow",
    },
    "comparison_card": {
        "structure": "Card -> Column -> [Text(h2), Row -> [multiple Column], Text]",
        "builder": "_build_comparison_card",
    },
    "nutrition_card": {
        "structure": "Card -> Column -> [Text(h3), Divider, multiple Row -> [Text, Text]]",
        "builder": "_build_nutrition_card",
    },
    "price_calculator": {
        "structure": "Card -> Column -> [Text, List -> [multiple Row], Divider, Button]",
        "builder": "_build_price_calculator",
    },
    "task_card": {
        "structure": "Card -> Column -> [Text(h2), Text, Row -> [Button, Button]]",
        "builder": "_build_task_card",
    },
    "step_card": {
        "structure": "Card -> Column -> [Text(h2), List -> [multiple Row]]",
        "builder": "_build_step_card",
    },
}

# A2UI v0.9 atomic component types
A2UI_COMPONENT_TYPES = [
    "Card",
    "Row",
    "Column",
    "Text",
    "Image",
    "Icon",
    "Button",
    "Divider",
    "List",
    "Spacer",
    "ProgressBar",
    "Badge",
    "Avatar",
    "Chip",
]

# Icon mapping for common use cases
ICON_MAPPING = {
    "navigation": "navigation",
    "location": "place",
    "building": "business",
    "bike": "directions_bike",
    "walk": "directions_walk",
    "car": "directions_car",
    "arrow_left": "arrow_back",
    "arrow_right": "arrow_forward",
    "arrow_forward": "arrow_upward",
    "arrow_back": "arrow_downward",
    "food": "restaurant",
    "shopping": "shopping_cart",
    "price": "attach_money",
    "nutrition": "eco",
    "compare": "compare_arrows",
    "info": "info",
    "warning": "warning",
    "success": "check_circle",
    "error": "error",
}


class A2UIConverter:
    """Convert business components to A2UI v0.9 standard atomic components.

    This converter transforms high-level business component definitions
    (like ar_label, map_card) into A2UI v0.9 compliant component trees
    built from atomic components (Card, Row, Text, etc.).

    Attributes:
        preserve_metadata: Whether to preserve original component metadata.
        include_visual_anchor: Whether to include visual anchor info in output.
    """

    def __init__(
        self,
        preserve_metadata: bool = True,
        include_visual_anchor: bool = True,
    ):
        """Initialize the converter.

        Args:
            preserve_metadata: Whether to preserve original metadata.
            include_visual_anchor: Whether to include AR anchor information.
        """
        self.preserve_metadata = preserve_metadata
        self.include_visual_anchor = include_visual_anchor

    def convert(self, component: dict) -> dict:
        """Convert a business component to A2UI v0.9 format.

        Args:
            component: The business component dictionary with type, id, and props.

        Returns:
            A2UI v0.9 compliant component tree.

        Raises:
            ValueError: If component type is not supported.
        """
        component_type = component.get("type")
        if not component_type:
            raise ValueError("Component must have a 'type' field")

        # Check if already in A2UI format
        if component_type in A2UI_COMPONENT_TYPES:
            logger.debug(f"Component {component_type} is already A2UI format")
            return component

        # Get the builder method
        mapping = COMPONENT_MAPPING.get(component_type)
        if not mapping:
            logger.warning(f"No mapping found for component type: {component_type}")
            return self._build_generic_card(component)

        builder_name = mapping["builder"]
        builder = getattr(self, builder_name, None)

        if not builder:
            logger.error(f"Builder method not found: {builder_name}")
            return self._build_generic_card(component)

        # Build the A2UI component
        a2ui_component = builder(component)

        # Add metadata if requested
        if self.preserve_metadata and "metadata" in component:
            a2ui_component["metadata"] = component["metadata"]

        # Add visual anchor if present and requested
        if self.include_visual_anchor and "visual_anchor" in component:
            a2ui_component["visual_anchor"] = component["visual_anchor"]

        return a2ui_component

    def _generate_id(self, prefix: str = "comp") -> str:
        """Generate a unique component ID."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _build_ar_label(self, component: dict) -> dict:
        """Build A2UI structure for ar_label.

        Structure: Card -> Row -> [Icon, Column -> [Text(h3), Text(caption)]]
        """
        props = component.get("props", {})

        # Build inner column with text elements
        text_column = {
            "type": "Column",
            "id": self._generate_id("col"),
            "props": {
                "crossAxisAlignment": "start",
            },
            "children": [
                {
                    "type": "Text",
                    "id": self._generate_id("txt"),
                    "props": {
                        "text": props.get("text", "Label"),
                        "variant": "h3",
                    },
                },
            ],
        }

        # Add subtext if present
        if props.get("subtext"):
            text_column["children"].append({
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": props["subtext"],
                    "variant": "caption",
                    "color": "secondary",
                },
            })

        # Build the row with icon and text
        row_children = []

        # Add icon if present
        icon_name = props.get("icon", "info")
        icon_name = ICON_MAPPING.get(icon_name, icon_name)
        row_children.append({
            "type": "Icon",
            "id": self._generate_id("icon"),
            "props": {
                "name": icon_name,
                "size": "medium",
            },
        })

        row_children.append(text_column)

        return {
            "type": "Card",
            "id": component.get("id", self._generate_id("ar_label")),
            "props": {
                "variant": "floating",
                "elevation": 2,
            },
            "children": [
                {
                    "type": "Row",
                    "id": self._generate_id("row"),
                    "props": {
                        "spacing": 8,
                        "crossAxisAlignment": "center",
                    },
                    "children": row_children,
                },
            ],
        }

    def _build_map_card(self, component: dict) -> dict:
        """Build A2UI structure for map_card.

        Structure: Card -> Column -> [Image, Text, markers..., Button]
        """
        props = component.get("props", {})

        children = []

        # Map image placeholder
        children.append({
            "type": "Image",
            "id": self._generate_id("img"),
            "props": {
                "src": props.get("map_url", "placeholder://map"),
                "alt": "Map view",
                "aspectRatio": "16:9",
                "fit": "cover",
            },
        })

        # Title
        children.append({
            "type": "Text",
            "id": self._generate_id("txt"),
            "props": {
                "text": props.get("title", "Map"),
                "variant": "h3",
            },
        })

        # Subtitle if present
        if props.get("subtitle"):
            children.append({
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": props["subtitle"],
                    "variant": "body2",
                    "color": "secondary",
                },
            })

        # Markers as list items
        markers = props.get("markers", [])
        for marker in markers:
            children.append({
                "type": "Row",
                "id": self._generate_id("row"),
                "props": {
                    "spacing": 8,
                },
                "children": [
                    {
                        "type": "Icon",
                        "id": self._generate_id("icon"),
                        "props": {
                            "name": ICON_MAPPING.get(marker.get("type", "location"), "place"),
                            "size": "small",
                        },
                    },
                    {
                        "type": "Text",
                        "id": self._generate_id("txt"),
                        "props": {
                            "text": f"{marker.get('label', '')} - {marker.get('distance', '')}",
                            "variant": "body2",
                        },
                    },
                ],
            })

        # Action button if present
        action = props.get("action")
        if action:
            children.append({
                "type": "Button",
                "id": self._generate_id("btn"),
                "props": {
                    "label": action.get("label", "Go"),
                    "variant": "primary",
                    "action": action.get("type", "navigate"),
                },
            })

        return {
            "type": "Card",
            "id": component.get("id", self._generate_id("map_card")),
            "props": {
                "variant": "elevated",
                "padding": 16,
            },
            "children": [
                {
                    "type": "Column",
                    "id": self._generate_id("col"),
                    "props": {
                        "spacing": 12,
                    },
                    "children": children,
                },
            ],
        }

    def _build_direction_arrow(self, component: dict) -> dict:
        """Build A2UI structure for direction_arrow.

        Structure: Card -> Row -> [Icon(arrow), Column -> [Text(destination), Text(distance)]]
        """
        props = component.get("props", {})

        direction = props.get("direction", "forward")
        icon_name = ICON_MAPPING.get(f"arrow_{direction}", "arrow_forward")

        return {
            "type": "Card",
            "id": component.get("id", self._generate_id("dir_arrow")),
            "props": {
                "variant": "floating",
            },
            "children": [
                {
                    "type": "Row",
                    "id": self._generate_id("row"),
                    "props": {
                        "spacing": 12,
                        "crossAxisAlignment": "center",
                    },
                    "children": [
                        {
                            "type": "Icon",
                            "id": self._generate_id("icon"),
                            "props": {
                                "name": icon_name,
                                "size": "large",
                                "color": "primary",
                            },
                        },
                        {
                            "type": "Column",
                            "id": self._generate_id("col"),
                            "props": {
                                "crossAxisAlignment": "start",
                            },
                            "children": [
                                {
                                    "type": "Text",
                                    "id": self._generate_id("txt"),
                                    "props": {
                                        "text": props.get("destination", "Destination"),
                                        "variant": "h3",
                                    },
                                },
                                {
                                    "type": "Text",
                                    "id": self._generate_id("txt"),
                                    "props": {
                                        "text": f"{props.get('distance', '')} · {props.get('eta', '')}",
                                        "variant": "caption",
                                        "color": "secondary",
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
        }

    def _build_comparison_card(self, component: dict) -> dict:
        """Build A2UI structure for comparison_card.

        Structure: Card -> Column -> [Text(h2), Row -> [multiple Column], Text]
        """
        props = component.get("props", {})

        # Build comparison columns
        items = props.get("items", [])
        item_columns = []

        for item in items:
            column_children = [
                {
                    "type": "Text",
                    "id": self._generate_id("txt"),
                    "props": {
                        "text": item.get("label", "Item"),
                        "variant": "subtitle1",
                        "weight": "bold" if item.get("highlight") else "normal",
                    },
                },
            ]

            # Add value or score
            if "value" in item:
                column_children.append({
                    "type": "Text",
                    "id": self._generate_id("txt"),
                    "props": {
                        "text": str(item["value"]),
                        "variant": "body1",
                    },
                })
            elif "score" in item:
                column_children.append({
                    "type": "Text",
                    "id": self._generate_id("txt"),
                    "props": {
                        "text": f"Score: {item['score']}",
                        "variant": "body1",
                        "color": "primary" if item.get("highlight") else "default",
                    },
                })

            item_columns.append({
                "type": "Column",
                "id": self._generate_id("col"),
                "props": {
                    "crossAxisAlignment": "center",
                    "flex": 1,
                },
                "children": column_children,
            })

        children = [
            {
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": props.get("title", "Comparison"),
                    "variant": "h2",
                },
            },
            {
                "type": "Row",
                "id": self._generate_id("row"),
                "props": {
                    "spacing": 16,
                    "mainAxisAlignment": "spaceEvenly",
                },
                "children": item_columns,
            },
        ]

        # Add recommendation if present
        if props.get("recommendation"):
            children.append({
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": props["recommendation"],
                    "variant": "body2",
                    "color": "success",
                },
            })

        return {
            "type": "Card",
            "id": component.get("id", self._generate_id("comp_card")),
            "props": {
                "variant": "elevated",
                "padding": 16,
            },
            "children": [
                {
                    "type": "Column",
                    "id": self._generate_id("col"),
                    "props": {
                        "spacing": 16,
                    },
                    "children": children,
                },
            ],
        }

    def _build_nutrition_card(self, component: dict) -> dict:
        """Build A2UI structure for nutrition_card.

        Structure: Card -> Column -> [Text(h3), Divider, multiple Row -> [Text, Text]]
        """
        props = component.get("props", {})

        children = [
            {
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": props.get("product_name", "Product"),
                    "variant": "h3",
                },
            },
            {
                "type": "Divider",
                "id": self._generate_id("div"),
                "props": {},
            },
        ]

        # Nutrition rows
        nutrition_fields = [
            ("calories", "Calories", "kcal"),
            ("protein", "Protein", ""),
            ("carbs", "Carbs", ""),
            ("fat", "Fat", ""),
            ("sugar", "Sugar", ""),
        ]

        for field, label, unit in nutrition_fields:
            value = props.get(field)
            if value is not None:
                display_value = f"{value}{unit}" if unit and isinstance(value, (int, float)) else str(value)
                children.append({
                    "type": "Row",
                    "id": self._generate_id("row"),
                    "props": {
                        "mainAxisAlignment": "spaceBetween",
                    },
                    "children": [
                        {
                            "type": "Text",
                            "id": self._generate_id("txt"),
                            "props": {
                                "text": label,
                                "variant": "body2",
                            },
                        },
                        {
                            "type": "Text",
                            "id": self._generate_id("txt"),
                            "props": {
                                "text": display_value,
                                "variant": "body2",
                                "weight": "bold",
                            },
                        },
                    ],
                })

        # Health rating if present
        if props.get("health_rating"):
            children.append({
                "type": "Divider",
                "id": self._generate_id("div"),
                "props": {},
            })
            children.append({
                "type": "Row",
                "id": self._generate_id("row"),
                "props": {
                    "mainAxisAlignment": "center",
                },
                "children": [
                    {
                        "type": "Badge",
                        "id": self._generate_id("badge"),
                        "props": {
                            "text": f"Health Rating: {props['health_rating']}/5",
                            "variant": "success" if props["health_rating"] >= 4 else "warning",
                        },
                    },
                ],
            })

        return {
            "type": "Card",
            "id": component.get("id", self._generate_id("nutr_card")),
            "props": {
                "variant": "elevated",
                "padding": 16,
            },
            "children": [
                {
                    "type": "Column",
                    "id": self._generate_id("col"),
                    "props": {
                        "spacing": 8,
                    },
                    "children": children,
                },
            ],
        }

    def _build_price_calculator(self, component: dict) -> dict:
        """Build A2UI structure for price_calculator.

        Structure: Card -> Column -> [Text, List -> [multiple Row], Divider, Button]
        """
        props = component.get("props", {})

        children = []

        # Title
        if props.get("title"):
            children.append({
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": props["title"],
                    "variant": "h3",
                },
            })

        # Items list
        items = props.get("items", [])
        item_rows = []

        for item in items:
            item_rows.append({
                "type": "Row",
                "id": self._generate_id("row"),
                "props": {
                    "mainAxisAlignment": "spaceBetween",
                },
                "children": [
                    {
                        "type": "Column",
                        "id": self._generate_id("col"),
                        "props": {
                            "crossAxisAlignment": "start",
                        },
                        "children": [
                            {
                                "type": "Text",
                                "id": self._generate_id("txt"),
                                "props": {
                                    "text": item.get("name", "Item"),
                                    "variant": "body1",
                                },
                            },
                            {
                                "type": "Text",
                                "id": self._generate_id("txt"),
                                "props": {
                                    "text": f"¥{item.get('price', 0)}/{item.get('unit', 'unit')}",
                                    "variant": "caption",
                                    "color": "secondary",
                                },
                            },
                        ],
                    },
                    {
                        "type": "Text",
                        "id": self._generate_id("txt"),
                        "props": {
                            "text": f"¥{item.get('unit_price', item.get('price', 0))}/unit",
                            "variant": "body1",
                            "weight": "bold",
                        },
                    },
                ],
            })

        if item_rows:
            children.append({
                "type": "List",
                "id": self._generate_id("list"),
                "props": {
                    "spacing": 12,
                },
                "children": item_rows,
            })

        # Total if present
        if props.get("total"):
            children.append({
                "type": "Divider",
                "id": self._generate_id("div"),
                "props": {},
            })
            children.append({
                "type": "Row",
                "id": self._generate_id("row"),
                "props": {
                    "mainAxisAlignment": "spaceBetween",
                },
                "children": [
                    {
                        "type": "Text",
                        "id": self._generate_id("txt"),
                        "props": {
                            "text": "Total",
                            "variant": "subtitle1",
                        },
                    },
                    {
                        "type": "Text",
                        "id": self._generate_id("txt"),
                        "props": {
                            "text": f"¥{props['total']}",
                            "variant": "h3",
                            "color": "primary",
                        },
                    },
                ],
            })

        # Recommendation if present
        if props.get("recommendation"):
            children.append({
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": props["recommendation"],
                    "variant": "body2",
                    "color": "success",
                },
            })

        return {
            "type": "Card",
            "id": component.get("id", self._generate_id("price_calc")),
            "props": {
                "variant": "elevated",
                "padding": 16,
            },
            "children": [
                {
                    "type": "Column",
                    "id": self._generate_id("col"),
                    "props": {
                        "spacing": 12,
                    },
                    "children": children,
                },
            ],
        }

    def _build_task_card(self, component: dict) -> dict:
        """Build A2UI structure for task_card."""
        props = component.get("props", {})

        children = [
            {
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": props.get("title", "Task"),
                    "variant": "h2",
                },
            },
        ]

        if props.get("description"):
            children.append({
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": props["description"],
                    "variant": "body1",
                },
            })

        if props.get("due_time"):
            children.append({
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": f"Due: {props['due_time']}",
                    "variant": "caption",
                    "color": "secondary",
                },
            })

        # Action buttons
        action = props.get("action", {})
        if action:
            children.append({
                "type": "Row",
                "id": self._generate_id("row"),
                "props": {
                    "spacing": 8,
                    "mainAxisAlignment": "end",
                },
                "children": [
                    {
                        "type": "Button",
                        "id": self._generate_id("btn"),
                        "props": {
                            "label": action.get("dismiss_label", "Cancel"),
                            "variant": "text",
                        },
                    },
                    {
                        "type": "Button",
                        "id": self._generate_id("btn"),
                        "props": {
                            "label": action.get("confirm_label", "Confirm"),
                            "variant": "primary",
                        },
                    },
                ],
            })

        return {
            "type": "Card",
            "id": component.get("id", self._generate_id("task_card")),
            "props": {
                "variant": "elevated",
                "padding": 16,
            },
            "children": [
                {
                    "type": "Column",
                    "id": self._generate_id("col"),
                    "props": {
                        "spacing": 12,
                    },
                    "children": children,
                },
            ],
        }

    def _build_step_card(self, component: dict) -> dict:
        """Build A2UI structure for step_card."""
        props = component.get("props", {})

        children = [
            {
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": props.get("title", "Steps"),
                    "variant": "h2",
                },
            },
        ]

        # Steps list
        steps = props.get("steps", [])
        current_step = props.get("current_step", 1)

        step_rows = []
        for step in steps:
            step_num = step.get("number", 0)
            is_completed = step.get("completed", False)
            is_current = step_num == current_step

            step_rows.append({
                "type": "Row",
                "id": self._generate_id("row"),
                "props": {
                    "spacing": 12,
                    "crossAxisAlignment": "center",
                },
                "children": [
                    {
                        "type": "Icon",
                        "id": self._generate_id("icon"),
                        "props": {
                            "name": "check_circle" if is_completed else ("radio_button_checked" if is_current else "radio_button_unchecked"),
                            "color": "success" if is_completed else ("primary" if is_current else "disabled"),
                        },
                    },
                    {
                        "type": "Text",
                        "id": self._generate_id("txt"),
                        "props": {
                            "text": step.get("instruction", "Step"),
                            "variant": "body1",
                            "color": "disabled" if is_completed else "default",
                        },
                    },
                ],
            })

        if step_rows:
            children.append({
                "type": "List",
                "id": self._generate_id("list"),
                "props": {
                    "spacing": 8,
                },
                "children": step_rows,
            })

        return {
            "type": "Card",
            "id": component.get("id", self._generate_id("step_card")),
            "props": {
                "variant": "elevated",
                "padding": 16,
            },
            "children": [
                {
                    "type": "Column",
                    "id": self._generate_id("col"),
                    "props": {
                        "spacing": 12,
                    },
                    "children": children,
                },
            ],
        }

    def _build_generic_card(self, component: dict) -> dict:
        """Build a generic card for unknown component types."""
        props = component.get("props", {})

        children = []

        # Try to extract common properties
        if props.get("title"):
            children.append({
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": props["title"],
                    "variant": "h3",
                },
            })

        if props.get("text"):
            children.append({
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": props["text"],
                    "variant": "body1",
                },
            })

        if props.get("description"):
            children.append({
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": props["description"],
                    "variant": "body2",
                },
            })

        # Fallback if no children
        if not children:
            children.append({
                "type": "Text",
                "id": self._generate_id("txt"),
                "props": {
                    "text": str(props) if props else "Unknown component",
                    "variant": "body2",
                },
            })

        return {
            "type": "Card",
            "id": component.get("id", self._generate_id("generic")),
            "props": {
                "variant": "outlined",
                "padding": 16,
            },
            "children": [
                {
                    "type": "Column",
                    "id": self._generate_id("col"),
                    "props": {
                        "spacing": 8,
                    },
                    "children": children,
                },
            ],
        }
