"""
A2UI Output Validator

Validates and normalizes UI components to ensure compatibility with A2UI 0.8 Preview.
"""

import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Valid A2UI 0.8 component types
VALID_COMPONENT_TYPES = {
    "Card",
    "Text",
    "Button",
    "Icon",
    "Badge",
    "Row",
    "Column",
    "Image",
    "List",
}

# Container types that should have children
CONTAINER_TYPES = {"Card", "Row", "Column", "List"}

# Required fields for A2UI components
REQUIRED_FIELDS = ["type", "props"]

# Recommended fields for full compatibility
RECOMMENDED_FIELDS = ["id", "children"]


def validate_a2ui_component(component: dict) -> tuple[bool, str]:
    """Validate a component against A2UI 0.8 specification.

    Args:
        component: The component dictionary to validate.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if not isinstance(component, dict):
        return False, f"Component must be a dict, got {type(component).__name__}"

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in component:
            return False, f"Missing required field: {field}"

    # Check type is valid
    comp_type = component.get("type")
    if comp_type not in VALID_COMPONENT_TYPES:
        return False, f"Unknown component type: {comp_type}. Valid types: {VALID_COMPONENT_TYPES}"

    # Check props is a dict
    if not isinstance(component.get("props"), dict):
        return False, f"props must be a dict, got {type(component.get('props')).__name__}"

    # Validate children if present
    if "children" in component:
        children = component["children"]
        if not isinstance(children, list):
            return False, f"children must be a list, got {type(children).__name__}"

        # Recursively validate children
        for i, child in enumerate(children):
            is_valid, error = validate_a2ui_component(child)
            if not is_valid:
                return False, f"Invalid child at index {i}: {error}"

    return True, ""


def normalize_component(component: dict) -> dict:
    """Normalize a component to A2UI 0.8 standard format.

    Adds missing id and children fields, ensures proper structure.

    Args:
        component: The component to normalize.

    Returns:
        Normalized component dict.
    """
    if not isinstance(component, dict):
        logger.warning(f"Cannot normalize non-dict component: {type(component)}")
        return component

    normalized = component.copy()

    # Ensure type field exists
    if "type" not in normalized:
        normalized["type"] = "Card"  # Default container type
        logger.warning("Component missing 'type' field, defaulting to 'Card'")

    # Ensure id field exists
    if "id" not in normalized:
        comp_type = normalized.get("type", "component")
        normalized["id"] = f"{comp_type.lower()}_{uuid.uuid4().hex[:8]}"

    # Ensure props field exists
    if "props" not in normalized:
        normalized["props"] = {}

    # Ensure children field exists for container types
    comp_type = normalized.get("type")
    if comp_type in CONTAINER_TYPES and "children" not in normalized:
        normalized["children"] = []

    # Recursively normalize children
    if "children" in normalized and isinstance(normalized["children"], list):
        normalized["children"] = [
            normalize_component(child) if isinstance(child, dict) else child
            for child in normalized["children"]
        ]

    return normalized


def move_visual_anchor_to_metadata(component: dict) -> dict:
    """Move visual_anchor from top-level to metadata for Preview compatibility.

    Args:
        component: Component that may contain visual_anchor.

    Returns:
        Component with visual_anchor moved to metadata.
    """
    if not isinstance(component, dict):
        return component

    result = component.copy()

    if "visual_anchor" in result:
        result["metadata"] = result.get("metadata", {})
        result["metadata"]["visual_anchor"] = result.pop("visual_anchor")
        logger.debug(f"Moved visual_anchor to metadata for component {result.get('id', 'unknown')}")

    # Recursively process children
    if "children" in result and isinstance(result["children"], list):
        result["children"] = [
            move_visual_anchor_to_metadata(child) if isinstance(child, dict) else child
            for child in result["children"]
        ]

    return result


def fix_trailing_commas(json_text: str) -> str:
    """Remove trailing commas from JSON text.

    Common issue with LLM-generated JSON.

    Args:
        json_text: Raw JSON text that may have trailing commas.

    Returns:
        JSON text with trailing commas removed.
    """
    # Remove trailing commas before closing braces/brackets
    fixed = re.sub(r',\s*([}\]])', r'\1', json_text)
    return fixed


def attempt_json_repair(json_text: str) -> str:
    """Attempt to repair common JSON issues.

    Handles:
    - Trailing commas
    - Missing closing brackets (truncated JSON)
    - Extra text after JSON

    Args:
        json_text: Malformed JSON text.

    Returns:
        Repaired JSON text (best effort).
    """
    text = json_text.strip()

    # Fix trailing commas
    text = fix_trailing_commas(text)

    # Count brackets to detect truncation
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')

    # Add missing closing brackets
    missing_braces = open_braces - close_braces
    missing_brackets = open_brackets - close_brackets

    if missing_braces > 0 or missing_brackets > 0:
        logger.warning(f"Attempting to repair truncated JSON: missing {missing_braces} braces, {missing_brackets} brackets")

        # Remove trailing incomplete content (after last complete value)
        # Find last complete structure point
        last_complete = max(
            text.rfind('}'),
            text.rfind(']'),
            text.rfind('"'),
        )

        if last_complete > 0:
            # Check if we're in the middle of a string
            if text[last_complete] == '"':
                # Find if this quote is escaped
                escaped = False
                i = last_complete - 1
                while i >= 0 and text[i] == '\\':
                    escaped = not escaped
                    i -= 1

                if not escaped:
                    # This is a closing quote, look for structure after
                    after_quote = text[last_complete + 1:].strip()
                    if after_quote and after_quote[0] not in ',}]:':
                        # Truncated after a string value
                        text = text[:last_complete + 1]

        # Add missing closures
        text += '}' * missing_braces
        text += ']' * missing_brackets

    return text


# Canonical prop key mappings: maps component type to (canonical_key, alternate_keys)
_PROP_KEY_CANONICALIZATION = {
    "Button": ("label", ["text", "content"]),
    "Text": ("content", ["label", "text"]),
    "Badge": ("text", ["label", "content"]),
}


def normalize_props(component: dict) -> dict:
    """Normalize text property keys to canonical forms for each component type.

    Different LLM providers use different keys for the same concept:
      - Button: canonical is "label", alternates are "text" and "content"
      - Text:   canonical is "content", alternates are "label" and "text"
      - Badge:  canonical is "text", alternates are "label" and "content"

    This function rewrites alternate keys to the canonical key so downstream
    renderers and consumers see a consistent schema.

    Args:
        component: A2UI component dict (modified in-place copy).

    Returns:
        Component with normalized prop keys.
    """
    if not isinstance(component, dict):
        return component

    result = component.copy()
    comp_type = result.get("type")
    props = result.get("props")

    if comp_type and isinstance(props, dict) and comp_type in _PROP_KEY_CANONICALIZATION:
        canonical_key, alt_keys = _PROP_KEY_CANONICALIZATION[comp_type]
        # Only act if the canonical key is missing or empty
        if not props.get(canonical_key):
            for alt in alt_keys:
                val = props.get(alt)
                if val:
                    props = props.copy()
                    props[canonical_key] = val
                    del props[alt]
                    logger.debug(
                        f"Normalized prop '{alt}' -> '{canonical_key}' "
                        f"for {comp_type} component {result.get('id', 'unknown')}"
                    )
                    break
            result["props"] = props

    # Recursively normalize children
    if "children" in result and isinstance(result["children"], list):
        result["children"] = [
            normalize_props(child) if isinstance(child, dict) else child
            for child in result["children"]
        ]

    return result


def validate_and_normalize(component: dict) -> tuple[dict, bool, str]:
    """Validate and normalize a component in one step.

    Args:
        component: The component to process.

    Returns:
        Tuple of (normalized_component, was_valid_before, error_if_still_invalid).
    """
    # First validate
    is_valid, error = validate_a2ui_component(component)

    # Normalize regardless
    normalized = normalize_component(component)

    # Move visual_anchor if present
    normalized = move_visual_anchor_to_metadata(normalized)

    # Re-validate after normalization
    is_valid_after, error_after = validate_a2ui_component(normalized)

    if not is_valid and is_valid_after:
        logger.info(f"Component normalized successfully: {error} -> valid")

    return normalized, is_valid, error_after if not is_valid_after else ""
