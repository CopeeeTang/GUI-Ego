"""
A2UI Preview Server

Simple HTTP server for previewing generated A2UI components in the browser.
Uses Python's built-in http.server for zero additional dependencies.
"""

import http.server
import json
import os
import socketserver
import urllib.parse
from pathlib import Path
from typing import Optional
import argparse
import logging

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "output"
DEFAULT_STATIC_PATH = Path(__file__).parent / "static"

# HTML template for rendering A2UI components
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A2UI Preview - {title}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }}

        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}

        .header h1 {{
            font-size: 24px;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            color: #888;
            font-size: 14px;
        }}

        .controls {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}

        .controls button {{
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}

        .controls button.primary {{
            background: #4CAF50;
            color: white;
        }}

        .controls button.secondary {{
            background: #2196F3;
            color: white;
        }}

        .controls button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}

        .preview-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }}

        .glasses-frame {{
            background: #000;
            border-radius: 20px;
            padding: 20px;
            width: 100%;
            max-width: 960px;
            aspect-ratio: 16/9;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}

        .glasses-frame::before {{
            content: '👓 Smart Glasses View';
            position: absolute;
            top: 10px;
            left: 20px;
            font-size: 12px;
            color: #666;
        }}

        .component-container {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 80%;
            max-height: 80%;
        }}

        /* A2UI Component Styles */
        .a2ui-card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            padding: 16px;
            color: #333;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            backdrop-filter: blur(10px);
        }}

        .a2ui-card.floating {{
            background: rgba(255, 255, 255, 0.9);
        }}

        .a2ui-row {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .a2ui-column {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .a2ui-text {{
            line-height: 1.4;
        }}

        .a2ui-text.h1 {{ font-size: 28px; font-weight: bold; }}
        .a2ui-text.h2 {{ font-size: 24px; font-weight: bold; }}
        .a2ui-text.h3 {{ font-size: 20px; font-weight: 600; }}
        .a2ui-text.subtitle1 {{ font-size: 16px; font-weight: 500; }}
        .a2ui-text.body1 {{ font-size: 16px; }}
        .a2ui-text.body2 {{ font-size: 14px; }}
        .a2ui-text.caption {{ font-size: 12px; color: #666; }}

        .a2ui-text.primary {{ color: #2196F3; }}
        .a2ui-text.secondary {{ color: #666; }}
        .a2ui-text.success {{ color: #4CAF50; }}

        .a2ui-icon {{
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }}

        .a2ui-icon.large {{ width: 32px; height: 32px; font-size: 28px; }}
        .a2ui-icon.small {{ width: 16px; height: 16px; font-size: 14px; }}

        .a2ui-button {{
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .a2ui-button.primary {{
            background: #2196F3;
            color: white;
        }}

        .a2ui-button.text {{
            background: transparent;
            color: #2196F3;
        }}

        .a2ui-divider {{
            height: 1px;
            background: #eee;
            width: 100%;
        }}

        .a2ui-badge {{
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }}

        .a2ui-badge.success {{
            background: #E8F5E9;
            color: #4CAF50;
        }}

        .a2ui-badge.warning {{
            background: #FFF3E0;
            color: #FF9800;
        }}

        .json-view {{
            background: #1e1e1e;
            border-radius: 12px;
            padding: 20px;
            width: 100%;
            max-width: 960px;
            overflow: auto;
            max-height: 400px;
        }}

        .json-view pre {{
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            color: #d4d4d4;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        .file-list {{
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            width: 100%;
            max-width: 960px;
        }}

        .file-list h3 {{
            margin-bottom: 15px;
        }}

        .file-list ul {{
            list-style: none;
        }}

        .file-list li {{
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 5px;
            background: rgba(255,255,255,0.05);
        }}

        .file-list a {{
            color: #4CAF50;
            text-decoration: none;
        }}

        .file-list a:hover {{
            text-decoration: underline;
        }}

        .metadata {{
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🕶️ A2UI Component Preview</h1>
        <p class="subtitle">{subtitle}</p>
    </div>

    <div class="controls">
        <button class="primary" onclick="location.href='/'">📁 File List</button>
        <button class="secondary" onclick="toggleJson()">📋 Toggle JSON</button>
        <button class="secondary" onclick="location.reload()">🔄 Refresh</button>
    </div>

    <div class="preview-container">
        <div class="glasses-frame">
            <div class="component-container" id="component-root">
                {component_html}
            </div>
        </div>

        <div class="json-view" id="json-view" style="display: none;">
            <pre>{json_content}</pre>
        </div>
    </div>

    <script>
        function toggleJson() {{
            const jsonView = document.getElementById('json-view');
            jsonView.style.display = jsonView.style.display === 'none' ? 'block' : 'none';
        }}
    </script>
</body>
</html>
"""

# File list template
FILE_LIST_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A2UI Preview - File Browser</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 40px 20px;
        }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        h1 {{ text-align: center; margin-bottom: 30px; }}
        .file-grid {{ display: grid; gap: 15px; }}
        .file-card {{
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.2s;
        }}
        .file-card:hover {{
            background: rgba(255,255,255,0.15);
            transform: translateY(-2px);
        }}
        .file-card a {{
            color: #4CAF50;
            text-decoration: none;
            font-size: 18px;
            font-weight: 500;
        }}
        .file-card .meta {{
            color: #888;
            font-size: 13px;
            margin-top: 8px;
        }}
        .empty {{
            text-align: center;
            color: #888;
            padding: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🕶️ A2UI Component Files</h1>
        <div class="file-grid">
            {file_cards}
        </div>
    </div>
</body>
</html>
"""

# Icon mapping for rendering
ICON_MAP = {
    "info": "ℹ️",
    "place": "📍",
    "navigation": "🧭",
    "directions_bike": "🚴",
    "directions_walk": "🚶",
    "directions_car": "🚗",
    "arrow_forward": "➡️",
    "arrow_back": "⬅️",
    "arrow_upward": "⬆️",
    "arrow_downward": "⬇️",
    "restaurant": "🍽️",
    "shopping_cart": "🛒",
    "attach_money": "💰",
    "eco": "🌿",
    "compare_arrows": "⚖️",
    "check_circle": "✅",
    "radio_button_checked": "🔘",
    "radio_button_unchecked": "⚪",
    "warning": "⚠️",
    "error": "❌",
    "business": "🏢",
}


def render_component(component: dict, depth: int = 0) -> str:
    """Render an A2UI component to HTML."""
    comp_type = component.get("type", "unknown")
    props = component.get("props", {})
    children = component.get("children", [])

    if comp_type == "Card":
        variant = props.get("variant", "elevated")
        inner = "".join(render_component(c, depth + 1) for c in children)
        return f'<div class="a2ui-card {variant}">{inner}</div>'

    elif comp_type == "Row":
        inner = "".join(render_component(c, depth + 1) for c in children)
        return f'<div class="a2ui-row">{inner}</div>'

    elif comp_type == "Column":
        inner = "".join(render_component(c, depth + 1) for c in children)
        return f'<div class="a2ui-column">{inner}</div>'

    elif comp_type == "Text":
        text = props.get("text", "")
        variant = props.get("variant", "body1")
        color = props.get("color", "")
        weight = props.get("weight", "")
        classes = f"a2ui-text {variant} {color}"
        style = f"font-weight: {weight};" if weight == "bold" else ""
        return f'<span class="{classes}" style="{style}">{text}</span>'

    elif comp_type == "Icon":
        name = props.get("name", "info")
        size = props.get("size", "medium")
        icon = ICON_MAP.get(name, "❓")
        return f'<span class="a2ui-icon {size}">{icon}</span>'

    elif comp_type == "Button":
        label = props.get("label", "Button")
        variant = props.get("variant", "primary")
        return f'<button class="a2ui-button {variant}">{label}</button>'

    elif comp_type == "Divider":
        return '<div class="a2ui-divider"></div>'

    elif comp_type == "Badge":
        text = props.get("text", "")
        variant = props.get("variant", "default")
        return f'<span class="a2ui-badge {variant}">{text}</span>'

    elif comp_type == "List":
        inner = "".join(render_component(c, depth + 1) for c in children)
        return f'<div class="a2ui-column">{inner}</div>'

    elif comp_type == "Image":
        alt = props.get("alt", "Image")
        return f'<div style="background:#ddd;padding:20px;border-radius:8px;text-align:center;">[{alt}]</div>'

    else:
        # Unknown component - render as JSON
        return f'<pre style="font-size:11px;color:#666;">{json.dumps(component, indent=2)}</pre>'


def render_legacy_component(component: dict) -> str:
    """Render a legacy (non-A2UI standard) component to HTML."""
    comp_type = component.get("type", "unknown")
    props = component.get("props", {})

    html = f'<div class="a2ui-card elevated">'
    html += f'<div class="a2ui-column">'

    # Render based on component type
    if comp_type == "ar_label":
        html += f'<div class="a2ui-row">'
        icon = ICON_MAP.get(props.get("icon", "info"), "ℹ️")
        html += f'<span class="a2ui-icon">{icon}</span>'
        html += f'<div class="a2ui-column">'
        html += f'<span class="a2ui-text h3">{props.get("text", "Label")}</span>'
        if props.get("subtext"):
            html += f'<span class="a2ui-text caption secondary">{props["subtext"]}</span>'
        html += '</div></div>'

    elif comp_type == "map_card":
        html += f'<span class="a2ui-text h3">{props.get("title", "Map")}</span>'
        if props.get("subtitle"):
            html += f'<span class="a2ui-text body2 secondary">{props["subtitle"]}</span>'
        for marker in props.get("markers", []):
            html += f'<div class="a2ui-row"><span class="a2ui-icon small">📍</span>'
            html += f'<span class="a2ui-text body2">{marker.get("label", "")} - {marker.get("distance", "")}</span></div>'
        if props.get("action"):
            html += f'<button class="a2ui-button primary">{props["action"].get("label", "Go")}</button>'

    elif comp_type == "comparison_card":
        html += f'<span class="a2ui-text h2">{props.get("title", "Comparison")}</span>'
        html += '<div class="a2ui-row" style="justify-content:space-around;">'
        for item in props.get("items", []):
            style = "font-weight:bold;" if item.get("highlight") else ""
            html += f'<div class="a2ui-column" style="align-items:center;{style}">'
            html += f'<span class="a2ui-text subtitle1">{item.get("label", "")}</span>'
            if "score" in item:
                html += f'<span class="a2ui-text body1">Score: {item["score"]}</span>'
            html += '</div>'
        html += '</div>'
        if props.get("recommendation"):
            html += f'<span class="a2ui-text body2 success">{props["recommendation"]}</span>'

    else:
        # Generic rendering
        for key, value in props.items():
            if isinstance(value, str):
                html += f'<span class="a2ui-text body2"><b>{key}:</b> {value}</span>'

    html += '</div></div>'
    return html


class PreviewHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for A2UI preview."""

    output_path: Path = DEFAULT_OUTPUT_PATH

    def do_GET(self):
        """Handle GET requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.serve_file_list()
        elif path == "/preview":
            file_name = query.get("file", [None])[0]
            if file_name:
                self.serve_preview(file_name)
            else:
                self.send_error(400, "Missing file parameter")
        elif path.startswith("/static/"):
            super().do_GET()
        else:
            self.send_error(404, "Not Found")

    def serve_file_list(self):
        """Serve the file browser page."""
        files = sorted(self.output_path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

        if not files:
            file_cards = '<div class="empty">No JSON files found in output directory</div>'
        else:
            cards = []
            for f in files[:20]:  # Limit to 20 most recent
                stat = f.stat()
                size_kb = stat.st_size / 1024
                from datetime import datetime
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                cards.append(f'''
                    <div class="file-card">
                        <a href="/preview?file={f.name}">{f.name}</a>
                        <div class="meta">{size_kb:.1f} KB • {mtime}</div>
                    </div>
                ''')
            file_cards = "".join(cards)

        html = FILE_LIST_TEMPLATE.format(file_cards=file_cards)
        self.send_html(html)

    def serve_preview(self, file_name: str):
        """Serve the component preview page."""
        file_path = self.output_path / file_name

        if not file_path.exists():
            self.send_error(404, f"File not found: {file_name}")
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
            return

        # Handle both single component and array of components
        if isinstance(data, list):
            components = data[:5]  # Limit to first 5
            subtitle = f"Showing {len(components)} of {len(data)} components from {file_name}"
        else:
            components = [data]
            subtitle = f"Single component from {file_name}"

        # Render components
        component_html = ""
        for i, comp in enumerate(components):
            if i > 0:
                component_html += '<div style="height:20px;"></div>'

            # Check if it's A2UI standard format (has "type" at root level that's an A2UI type)
            comp_type = comp.get("type", "")
            if comp_type in ["Card", "Row", "Column", "Text", "Icon", "Button"]:
                component_html += render_component(comp)
            else:
                component_html += render_legacy_component(comp)

        html = HTML_TEMPLATE.format(
            title=file_name,
            subtitle=subtitle,
            component_html=component_html,
            json_content=json.dumps(data, indent=2, ensure_ascii=False),
        )

        self.send_html(html)

    def send_html(self, html: str):
        """Send HTML response."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html.encode("utf-8")))
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        """Override to use logger."""
        logger.info(f"{self.address_string()} - {format % args}")


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    output_path: Optional[str] = None,
):
    """Run the preview server.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        output_path: Path to output directory containing JSON files.
    """
    if output_path:
        PreviewHandler.output_path = Path(output_path)
    else:
        PreviewHandler.output_path = DEFAULT_OUTPUT_PATH

    PreviewHandler.output_path.mkdir(parents=True, exist_ok=True)

    with socketserver.TCPServer((host, port), PreviewHandler) as httpd:
        print(f"🕶️  A2UI Preview Server")
        print(f"   URL: http://{host}:{port}")
        print(f"   Output path: {PreviewHandler.output_path}")
        print(f"   Press Ctrl+C to stop")
        print()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="A2UI Preview Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--output-path", "-o", help="Path to output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(message)s")

    run_server(
        host=args.host,
        port=args.port,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
