"""UI rendering using headless browser.

Renders A2UI JSON components to images using Playwright,
ensuring visual consistency with the preview server.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Check if Playwright is available
try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning(
        "Playwright not installed. Install with: "
        "pip install playwright && playwright install chromium"
    )


class UIRenderer:
    """Render UI components to images using headless browser.

    Uses Playwright to load the preview server page and capture
    screenshots of rendered UI components with transparent backgrounds.

    Attributes:
        preview_server_url: URL of the running preview server.
        viewport_size: Tuple of (width, height) for the browser viewport.
    """

    def __init__(
        self,
        preview_server_url: str = "http://localhost:8080",
        viewport_size: tuple[int, int] = (1280, 720),
    ):
        """Initialize the UI renderer.

        Args:
            preview_server_url: URL of the A2UI preview server.
            viewport_size: Browser viewport dimensions (width, height).
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright is required for UI rendering. "
                "Install with: pip install playwright && playwright install chromium"
            )

        self.preview_server_url = preview_server_url.rstrip("/")
        self.viewport_size = viewport_size
        self._browser: Optional[Browser] = None
        self._playwright = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._start_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._stop_browser()

    async def _start_browser(self) -> None:
        """Start the headless browser."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=["--disable-gpu", "--no-sandbox"],
        )
        logger.info("Started headless Chromium browser")

    async def _stop_browser(self) -> None:
        """Stop the headless browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Stopped headless browser")

    async def render_to_image(
        self,
        ui_json_path: Path | str,
        transparent_bg: bool = True,
    ) -> np.ndarray:
        """Render a UI JSON file to an RGBA image.

        Args:
            ui_json_path: Path to the UI JSON file.
            transparent_bg: Whether to use transparent background.

        Returns:
            RGBA numpy array of the rendered UI.

        Raises:
            RuntimeError: If browser is not started.
            FileNotFoundError: If the JSON file doesn't exist.
        """
        if not self._browser:
            raise RuntimeError("Browser not started. Use async context manager.")

        ui_json_path = Path(ui_json_path)
        if not ui_json_path.exists():
            raise FileNotFoundError(f"UI JSON not found: {ui_json_path}")

        # Create a new page
        page = await self._browser.new_page(
            viewport={"width": self.viewport_size[0], "height": self.viewport_size[1]},
        )

        try:
            # Build the preview URL
            # The path needs to be relative to the output directory
            relative_path = ui_json_path.name
            if ui_json_path.parent.name != "output":
                # Try to build relative path from output directory
                try:
                    parts = ui_json_path.parts
                    output_idx = parts.index("output") if "output" in parts else -1
                    if output_idx >= 0:
                        relative_path = "/".join(parts[output_idx + 1:])
                except (ValueError, IndexError):
                    relative_path = ui_json_path.name

            preview_url = f"{self.preview_server_url}/preview?file={relative_path}"
            logger.debug(f"Loading preview URL: {preview_url}")

            # Navigate to the preview page
            await page.goto(preview_url, wait_until="networkidle")

            # Inject CSS for transparent background if needed
            if transparent_bg:
                await page.add_style_tag(content="""
                    body { background: transparent !important; }
                    .glasses-frame {
                        background: transparent !important;
                        box-shadow: none !important;
                    }
                    .glasses-frame::after { display: none !important; }
                    .header, .controls, .json-view { display: none !important; }
                """)

            # Wait for components to render
            await page.wait_for_timeout(500)

            # Find the component container
            component_container = await page.query_selector(".component-container")
            if not component_container:
                component_container = await page.query_selector(".glasses-frame")

            if not component_container:
                logger.warning("Component container not found, capturing full page")
                screenshot_bytes = await page.screenshot(type="png")
            else:
                # Get bounding box and capture just the component
                screenshot_bytes = await component_container.screenshot(
                    type="png",
                    omit_background=transparent_bg,
                )

            # Convert to numpy array
            import io
            from PIL import Image

            image = Image.open(io.BytesIO(screenshot_bytes))
            if transparent_bg and image.mode != "RGBA":
                image = image.convert("RGBA")

            img_array = np.array(image)
            logger.info(f"Rendered UI image: {img_array.shape}")

            return img_array

        finally:
            await page.close()

    async def render_from_json(
        self,
        ui_data: dict,
        output_path: Optional[Path] = None,
        transparent_bg: bool = True,
    ) -> np.ndarray:
        """Render UI from JSON data directly.

        Creates a temporary file and renders it.

        Args:
            ui_data: UI component dictionary.
            output_path: Optional path to save temporary JSON.
            transparent_bg: Whether to use transparent background.

        Returns:
            RGBA numpy array of the rendered UI.
        """
        import tempfile

        # Create temporary JSON file
        if output_path:
            temp_path = output_path
        else:
            fd, temp_path = tempfile.mkstemp(suffix=".json")
            temp_path = Path(temp_path)

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(ui_data, f, ensure_ascii=False, indent=2)

            return await self.render_to_image(temp_path, transparent_bg)

        finally:
            if not output_path and temp_path.exists():
                temp_path.unlink()


def render_ui_sync(
    ui_json_path: Path | str,
    preview_server_url: str = "http://localhost:8080",
    viewport_size: tuple[int, int] = (1280, 720),
    transparent_bg: bool = True,
) -> np.ndarray:
    """Synchronous wrapper for UI rendering.

    Args:
        ui_json_path: Path to the UI JSON file.
        preview_server_url: URL of the preview server.
        viewport_size: Browser viewport dimensions.
        transparent_bg: Whether to use transparent background.

    Returns:
        RGBA numpy array of the rendered UI.
    """
    async def _render():
        async with UIRenderer(preview_server_url, viewport_size) as renderer:
            return await renderer.render_to_image(ui_json_path, transparent_bg)

    return asyncio.run(_render())
