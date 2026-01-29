"""A2UI message builder for constructing complete A2UI message sequences."""

import logging
import uuid
from typing import Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class A2UIMessageBuilder:
    """Build complete A2UI v0.9 message sequences.

    A2UI communication consists of message sequences that:
    1. Create surfaces (display areas)
    2. Update components on surfaces
    3. Update data models for dynamic content

    This builder creates properly structured message sequences
    that can be consumed by A2UI renderers.
    """

    # Default catalog for smart glasses components
    DEFAULT_CATALOG_ID = "smart-glasses-ui-v1"

    def __init__(
        self,
        catalog_id: Optional[str] = None,
        include_timestamps: bool = True,
    ):
        """Initialize the message builder.

        Args:
            catalog_id: The component catalog ID to use.
            include_timestamps: Whether to include timestamps in messages.
        """
        self.catalog_id = catalog_id or self.DEFAULT_CATALOG_ID
        self.include_timestamps = include_timestamps

    def _generate_surface_id(self) -> str:
        """Generate a unique surface ID."""
        return f"surface_{uuid.uuid4().hex[:12]}"

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.utcnow().isoformat() + "Z"

    def build_messages(
        self,
        components: list[dict],
        surface_id: Optional[str] = None,
        data_model: Optional[dict] = None,
    ) -> list[dict]:
        """Build a complete A2UI message sequence.

        Creates:
        1. createSurface message
        2. updateComponents message
        3. updateDataModel message (if data_model provided)

        Args:
            components: List of A2UI components to display.
            surface_id: Optional surface ID. Generated if not provided.
            data_model: Optional data model for dynamic content.

        Returns:
            List of A2UI messages in order.
        """
        surface_id = surface_id or self._generate_surface_id()
        messages = []

        # 1. Create surface
        messages.append(self._build_create_surface(surface_id))

        # 2. Update components
        messages.append(self._build_update_components(surface_id, components))

        # 3. Update data model (if provided)
        if data_model:
            messages.append(self._build_update_data_model(surface_id, data_model))

        return messages

    def build_single_component_messages(
        self,
        component: dict,
        surface_id: Optional[str] = None,
    ) -> list[dict]:
        """Build messages for a single component.

        Convenience method for single component displays.

        Args:
            component: The A2UI component.
            surface_id: Optional surface ID.

        Returns:
            List of A2UI messages.
        """
        return self.build_messages([component], surface_id)

    def _build_create_surface(self, surface_id: str) -> dict:
        """Build a createSurface message."""
        message = {
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": self.catalog_id,
                "type": "overlay",  # Smart glasses typically use overlay surfaces
                "properties": {
                    "anchor": "world",  # AR anchor to world
                    "persistent": False,
                    "interactive": True,
                },
            },
        }

        if self.include_timestamps:
            message["timestamp"] = self._get_timestamp()

        return message

    def _build_update_components(
        self,
        surface_id: str,
        components: list[dict],
    ) -> dict:
        """Build an updateComponents message."""
        message = {
            "updateComponents": {
                "surfaceId": surface_id,
                "components": components,
            },
        }

        if self.include_timestamps:
            message["timestamp"] = self._get_timestamp()

        return message

    def _build_update_data_model(
        self,
        surface_id: str,
        data: dict,
        path: str = "/",
    ) -> dict:
        """Build an updateDataModel message."""
        message = {
            "updateDataModel": {
                "surfaceId": surface_id,
                "path": path,
                "value": data,
            },
        }

        if self.include_timestamps:
            message["timestamp"] = self._get_timestamp()

        return message

    def build_destroy_surface(self, surface_id: str) -> dict:
        """Build a destroySurface message."""
        message = {
            "destroySurface": {
                "surfaceId": surface_id,
            },
        }

        if self.include_timestamps:
            message["timestamp"] = self._get_timestamp()

        return message

    def build_batch_update(
        self,
        updates: list[tuple[str, list[dict]]],
    ) -> list[dict]:
        """Build batch update messages for multiple surfaces.

        Args:
            updates: List of (surface_id, components) tuples.

        Returns:
            List of updateComponents messages.
        """
        messages = []
        for surface_id, components in updates:
            messages.append(self._build_update_components(surface_id, components))
        return messages


class A2UISession:
    """Manage an A2UI session with multiple surfaces.

    This class tracks active surfaces and provides methods
    for managing the session lifecycle.
    """

    def __init__(self, message_builder: Optional[A2UIMessageBuilder] = None):
        """Initialize the session.

        Args:
            message_builder: Message builder to use. Created if not provided.
        """
        self.builder = message_builder or A2UIMessageBuilder()
        self.active_surfaces: dict[str, dict] = {}
        self.message_history: list[dict] = []

    def create_surface(
        self,
        components: list[dict],
        surface_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> tuple[str, list[dict]]:
        """Create a new surface with components.

        Args:
            components: Components to display.
            surface_id: Optional surface ID.
            metadata: Optional metadata to track.

        Returns:
            Tuple of (surface_id, messages).
        """
        surface_id = surface_id or self.builder._generate_surface_id()

        messages = self.builder.build_messages(components, surface_id)

        # Track the surface
        self.active_surfaces[surface_id] = {
            "components": components,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }

        self.message_history.extend(messages)

        return surface_id, messages

    def update_surface(
        self,
        surface_id: str,
        components: list[dict],
    ) -> list[dict]:
        """Update components on an existing surface.

        Args:
            surface_id: The surface to update.
            components: New components.

        Returns:
            List of update messages.

        Raises:
            KeyError: If surface doesn't exist.
        """
        if surface_id not in self.active_surfaces:
            raise KeyError(f"Surface not found: {surface_id}")

        message = self.builder._build_update_components(surface_id, components)
        messages = [message]

        # Update tracking
        self.active_surfaces[surface_id]["components"] = components
        self.active_surfaces[surface_id]["updated_at"] = datetime.utcnow().isoformat()

        self.message_history.extend(messages)

        return messages

    def destroy_surface(self, surface_id: str) -> list[dict]:
        """Destroy a surface.

        Args:
            surface_id: The surface to destroy.

        Returns:
            List of destroy messages.
        """
        if surface_id in self.active_surfaces:
            del self.active_surfaces[surface_id]

        message = self.builder.build_destroy_surface(surface_id)
        messages = [message]

        self.message_history.extend(messages)

        return messages

    def destroy_all_surfaces(self) -> list[dict]:
        """Destroy all active surfaces.

        Returns:
            List of destroy messages.
        """
        messages = []
        for surface_id in list(self.active_surfaces.keys()):
            messages.extend(self.destroy_surface(surface_id))
        return messages

    def get_session_state(self) -> dict:
        """Get the current session state.

        Returns:
            Dictionary with session information.
        """
        return {
            "active_surfaces": len(self.active_surfaces),
            "surfaces": {
                sid: {
                    "component_count": len(info["components"]),
                    "created_at": info.get("created_at"),
                    "updated_at": info.get("updated_at"),
                }
                for sid, info in self.active_surfaces.items()
            },
            "total_messages": len(self.message_history),
        }

    def export_messages(self) -> list[dict]:
        """Export all messages in the session.

        Returns:
            Complete message history.
        """
        return self.message_history.copy()


def format_for_preview(messages: list[dict]) -> str:
    """Format A2UI messages for preview display.

    Args:
        messages: List of A2UI messages.

    Returns:
        JSON string formatted for readability.
    """
    import json
    return json.dumps(messages, indent=2, ensure_ascii=False)


def validate_message_sequence(messages: list[dict]) -> tuple[bool, list[str]]:
    """Validate an A2UI message sequence.

    Checks:
    - createSurface must come before updates to that surface
    - All required fields are present
    - Surface IDs are consistent

    Args:
        messages: List of A2UI messages.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []
    created_surfaces = set()

    for i, message in enumerate(messages):
        # Check createSurface
        if "createSurface" in message:
            create_msg = message["createSurface"]
            if "surfaceId" not in create_msg:
                errors.append(f"Message {i}: createSurface missing surfaceId")
            else:
                created_surfaces.add(create_msg["surfaceId"])

            if "catalogId" not in create_msg:
                errors.append(f"Message {i}: createSurface missing catalogId")

        # Check updateComponents
        elif "updateComponents" in message:
            update_msg = message["updateComponents"]
            if "surfaceId" not in update_msg:
                errors.append(f"Message {i}: updateComponents missing surfaceId")
            elif update_msg["surfaceId"] not in created_surfaces:
                errors.append(
                    f"Message {i}: updateComponents references unknown surface "
                    f"'{update_msg['surfaceId']}'"
                )

            if "components" not in update_msg:
                errors.append(f"Message {i}: updateComponents missing components")

        # Check updateDataModel
        elif "updateDataModel" in message:
            data_msg = message["updateDataModel"]
            if "surfaceId" not in data_msg:
                errors.append(f"Message {i}: updateDataModel missing surfaceId")
            elif data_msg["surfaceId"] not in created_surfaces:
                errors.append(
                    f"Message {i}: updateDataModel references unknown surface "
                    f"'{data_msg['surfaceId']}'"
                )

        # Check destroySurface
        elif "destroySurface" in message:
            destroy_msg = message["destroySurface"]
            if "surfaceId" not in destroy_msg:
                errors.append(f"Message {i}: destroySurface missing surfaceId")
            else:
                created_surfaces.discard(destroy_msg["surfaceId"])

    return len(errors) == 0, errors
