"""
A2UI Preview Server

Simple HTTP server for previewing generated A2UI components in the browser.
Uses Python's built-in http.server for zero additional dependencies.
"""

import http.server
import html
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

        .a2ui-badge.error {{
            background: #FFEBEE;
            color: #F44336;
        }}

        /* ProgressBar Styles */
        .a2ui-progress {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
            width: 100%;
        }}

        .a2ui-progress.a2ui-progress-default {{
            height: 8px;
        }}

        .a2ui-progress.a2ui-progress-slim {{
            height: 4px;
        }}

        .a2ui-progress-bar {{
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
            height: 100%;
            border-radius: 8px;
            transition: width 0.3s ease;
        }}

        /* Image Styles */
        .a2ui-image {{
            max-width: 100%;
            border-radius: 8px;
        }}

        .a2ui-image-placeholder {{
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            color: #666;
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
    <title>A2UI Preview - Lab Archives</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0e27;
            --bg-secondary: #151932;
            --bg-card: #1a1f3a;
            --accent-v1: #00d4ff;
            --accent-v2: #00ff88;
            --accent-v3: #bd00ff;
            --accent-other: #ff6b35;
            --text-primary: #e0e6ed;
            --text-secondary: #8b92a8;
            --text-muted: #4a5568;
            --border: rgba(255, 255, 255, 0.1);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}

        @keyframes slideIn {{
            from {{
                transform: translateX(-100%);
                opacity: 0;
            }}
            to {{
                transform: translateX(0);
                opacity: 1;
            }}
        }}

        body {{
            font-family: 'IBM Plex Mono', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 0;
            position: relative;
            overflow-x: hidden;
        }}

        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background:
                radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(189, 0, 255, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(0, 255, 136, 0.03) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 60px 40px;
            position: relative;
            z-index: 1;
        }}

        .header {{
            text-align: center;
            margin-bottom: 60px;
            animation: fadeInUp 0.6s ease-out;
        }}

        .header h1 {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 48px;
            font-weight: 700;
            margin-bottom: 12px;
            background: linear-gradient(135deg, var(--accent-v1), var(--accent-v3));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.02em;
        }}

        .header .subtitle {{
            font-size: 14px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.15em;
            font-weight: 500;
        }}

        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 24px;
            animation: fadeInUp 0.6s ease-out 0.1s backwards;
        }}

        .stat-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }}

        .stat-number {{
            font-weight: 600;
            color: var(--accent-v2);
            font-size: 18px;
        }}

        .stat-label {{
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}

        .version-groups {{
            display: flex;
            flex-direction: column;
            gap: 48px;
        }}

        .version-section {{
            animation: fadeInUp 0.6s ease-out backwards;
        }}

        .version-section:nth-child(1) {{ animation-delay: 0.2s; }}
        .version-section:nth-child(2) {{ animation-delay: 0.3s; }}
        .version-section:nth-child(3) {{ animation-delay: 0.4s; }}
        .version-section:nth-child(4) {{ animation-delay: 0.5s; }}

        .version-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }}

        .version-badge {{
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            position: relative;
            overflow: hidden;
        }}

        .version-badge::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }}

        .version-badge:hover::before {{
            left: 100%;
        }}

        .version-badge.v1 {{
            background: rgba(0, 212, 255, 0.15);
            color: var(--accent-v1);
            border: 1px solid rgba(0, 212, 255, 0.3);
        }}

        .version-badge.v2 {{
            background: rgba(0, 255, 136, 0.15);
            color: var(--accent-v2);
            border: 1px solid rgba(0, 255, 136, 0.3);
        }}

        .version-badge.v3 {{
            background: rgba(189, 0, 255, 0.15);
            color: var(--accent-v3);
            border: 1px solid rgba(189, 0, 255, 0.3);
        }}

        .version-badge.other {{
            background: rgba(255, 107, 53, 0.15);
            color: var(--accent-other);
            border: 1px solid rgba(255, 107, 53, 0.3);
        }}

        .version-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 20px;
            font-weight: 500;
            color: var(--text-primary);
        }}

        .version-count {{
            margin-left: auto;
            font-size: 12px;
            color: var(--text-muted);
        }}

        .file-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 16px;
        }}

        .file-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            cursor: pointer;
        }}

        .file-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 3px;
            height: 100%;
            background: var(--accent-v1);
            transform: scaleY(0);
            transition: transform 0.3s ease;
        }}

        .file-card.v1::before {{ background: var(--accent-v1); }}
        .file-card.v2::before {{ background: var(--accent-v2); }}
        .file-card.v3::before {{ background: var(--accent-v3); }}
        .file-card.other::before {{ background: var(--accent-other); }}

        .file-card:hover {{
            transform: translateX(8px);
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}

        .file-card:hover::before {{
            transform: scaleY(1);
        }}

        .file-header {{
            display: flex;
            align-items: start;
            gap: 12px;
            margin-bottom: 12px;
        }}

        .file-icon {{
            width: 40px;
            height: 40px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
            background: rgba(255, 255, 255, 0.05);
        }}

        .file-info {{
            flex: 1;
            min-width: 0;
        }}

        .file-name {{
            color: var(--text-primary);
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            display: block;
            word-break: break-word;
            line-height: 1.4;
            transition: color 0.2s;
        }}

        .file-card:hover .file-name {{
            color: var(--accent-v2);
        }}

        .file-meta {{
            display: flex;
            gap: 16px;
            margin-top: 12px;
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .meta-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .meta-icon {{
            opacity: 0.5;
        }}

        .empty {{
            text-align: center;
            color: var(--text-muted);
            padding: 80px 20px;
            animation: pulse 2s ease-in-out infinite;
        }}

        .empty-icon {{
            font-size: 64px;
            margin-bottom: 16px;
            opacity: 0.3;
        }}

        .search-bar {{
            max-width: 600px;
            margin: 0 auto 48px;
            position: relative;
            animation: fadeInUp 0.6s ease-out 0.15s backwards;
        }}

        .search-input {{
            width: 100%;
            padding: 16px 48px 16px 20px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--text-primary);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 14px;
            transition: all 0.3s;
        }}

        .search-input:focus {{
            outline: none;
            border-color: var(--accent-v2);
            box-shadow: 0 0 0 3px rgba(0, 255, 136, 0.1);
        }}

        .search-icon {{
            position: absolute;
            right: 16px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            pointer-events: none;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 40px 20px;
            }}

            .header h1 {{
                font-size: 32px;
            }}

            .stats {{
                flex-direction: column;
                gap: 16px;
            }}

            .file-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>A2UI LAB ARCHIVES</h1>
            <p class="subtitle">Component Output Browser</p>
            <div class="stats">
                <div class="stat-item">
                    <span class="stat-number" id="total-files">0</span>
                    <span class="stat-label">Files</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number" id="total-versions">0</span>
                    <span class="stat-label">Versions</span>
                </div>
            </div>
        </div>

        <div class="search-bar">
            <input
                type="text"
                class="search-input"
                placeholder="Search files by name or version..."
                id="search-input"
                autocomplete="off"
            >
            <span class="search-icon">🔍</span>
        </div>

        <div class="version-groups" id="version-groups">
            {version_sections}
        </div>
    </div>

    <script>
        // Search functionality
        const searchInput = document.getElementById('search-input');
        const versionGroups = document.getElementById('version-groups');

        searchInput.addEventListener('input', (e) => {{
            const query = e.target.value.toLowerCase();
            const sections = versionGroups.querySelectorAll('.version-section');

            sections.forEach(section => {{
                const cards = section.querySelectorAll('.file-card');
                let visibleCount = 0;

                cards.forEach(card => {{
                    const fileName = card.querySelector('.file-name').textContent.toLowerCase();
                    if (fileName.includes(query)) {{
                        card.style.display = 'block';
                        visibleCount++;
                    }} else {{
                        card.style.display = 'none';
                    }}
                }});

                section.style.display = visibleCount > 0 ? 'block' : 'none';
            }});
        }});

        // Update stats
        const totalFiles = document.querySelectorAll('.file-card').length;
        const totalVersions = document.querySelectorAll('.version-section').length;
        document.getElementById('total-files').textContent = totalFiles;
        document.getElementById('total-versions').textContent = totalVersions;
    </script>
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

# Variant CSS class mapping for semantic-to-style translation
# The renderer uses these classes instead of inline styles
# Variant CSS class mapping for semantic-to-style translation
VARIANT_CLASSES = {
    "Card": {
        "glass": "a2ui-card-glass",
        "solid": "a2ui-card-solid",
        "outline": "a2ui-card-outline",
        "alert": "a2ui-card-alert",
        # Legacy variants mapped to new system
        "floating": "a2ui-card-glass",
        "elevated": "a2ui-card-solid",
        "outlined": "a2ui-card-outline",
    },
    "Text": {
        "hero": "a2ui-text-hero",
        "h1": "a2ui-text-h1",
        "h2": "a2ui-text-h2",
        "h3": "a2ui-text-h2",  # Map h3 to h2
        "body": "a2ui-text-body",
        "body1": "a2ui-text-body",
        "body2": "a2ui-text-body",
        "caption": "a2ui-text-caption",
        "label": "a2ui-text-label",
        "subtitle1": "a2ui-text-h2",
    },
    "Button": {
        "primary": "a2ui-btn-primary",
        "secondary": "a2ui-btn-secondary",
        "ghost": "a2ui-btn-ghost",
        "icon_only": "a2ui-btn-icon",
        "text": "a2ui-btn-ghost",  # Legacy mapping
    },
    "Badge": {
        "info": "a2ui-badge-info",
        "success": "a2ui-badge-success",
        "warning": "a2ui-badge-warning",
        "error": "a2ui-badge-error",
        "default": "a2ui-badge-info",
    },
    "Icon": {
        "small": "a2ui-icon-sm",
        "medium": "a2ui-icon-md",
        "large": "a2ui-icon-lg",
    },
    "ProgressBar": {
        "default": "a2ui-progress-default",
        "slim": "a2ui-progress-slim",
    },
}

# Constants for security validation
ALLOWED_URL_SCHEMES = ('http://', 'https://', '/')
PERCENT_MIN = 0
PERCENT_MAX = 100


def escape_html(value: str) -> str:
    """Escape HTML content to prevent XSS attacks."""
    return html.escape(str(value))


def escape_html_attr(value: str) -> str:
    """Escape HTML attribute values to prevent XSS attacks."""
    return html.escape(str(value), quote=True)


def clamp_percentage(value: float, max_value: float) -> float:
    """Clamp a value to 0-100% range based on max value."""
    if max_value <= 0:
        return PERCENT_MIN
    percent = (value / max_value) * 100
    return max(PERCENT_MIN, min(PERCENT_MAX, percent))


def render_component(component: dict, depth: int = 0) -> str:
    """Render an A2UI component to HTML using variant classes (no inline styles)."""
    comp_type = component.get("type", "unknown")
    props = component.get("props", {})

    # Support both standard "children" and non-standard "content" fields
    children = component.get("children", [])
    if not children and "content" in component:
        content = component["content"]
        # If content is a single component, wrap it in a list
        children = [content] if isinstance(content, dict) else content

    if comp_type == "Card":
        variant = props.get("variant", "glass")
        variant_class = VARIANT_CLASSES["Card"].get(variant, "a2ui-card-glass")
        inner = "".join(render_component(c, depth + 1) for c in children)
        return f'<div class="a2ui-card {variant_class}">{inner}</div>'

    elif comp_type == "Row":
        inner = "".join(render_component(c, depth + 1) for c in children)
        return f'<div class="a2ui-row">{inner}</div>'

    elif comp_type == "Column":
        inner = "".join(render_component(c, depth + 1) for c in children)
        return f'<div class="a2ui-column">{inner}</div>'

    elif comp_type == "Text":
        # Support both "text" in props and "content" at root level
        text = props.get("text", "") or component.get("content", "")
        variant = props.get("variant", "body")
        variant_class = VARIANT_CLASSES["Text"].get(variant, "a2ui-text-body")
        text_escaped = escape_html(text)
        return f'<span class="a2ui-text {variant_class}">{text_escaped}</span>'

    elif comp_type == "Icon":
        # Extract icon properties
        name = props.get("name", "info")
        size = props.get("size", "medium")
        size_class = VARIANT_CLASSES["Icon"].get(size, "a2ui-icon-md")
        # Ensure name is a string for ICON_MAP
        if isinstance(name, dict):
            name = name.get("name", "info")
        icon = ICON_MAP.get(name, "❓")
        return f'<span class="a2ui-icon {size_class}">{icon}</span>'

    elif comp_type == "Button":
        label = props.get("label", "Button")
        variant = props.get("variant", "primary")
        variant_class = VARIANT_CLASSES["Button"].get(variant, "a2ui-btn-primary")
        label_escaped = escape_html(label)
        return f'<button class="a2ui-button {variant_class}">{label_escaped}</button>'

    elif comp_type == "Divider":
        return '<div class="a2ui-divider"></div>'

    elif comp_type == "Badge":
        text = props.get("text", "")
        variant = props.get("variant", "info")
        variant_class = VARIANT_CLASSES["Badge"].get(variant, "a2ui-badge-info")
        text_escaped = escape_html(text)
        return f'<span class="a2ui-badge {variant_class}">{text_escaped}</span>'

    elif comp_type == "ProgressBar":
        value = props.get("value", 0)
        max_val = props.get("max", 100)
        variant = props.get("variant", "default")
        variant_class = VARIANT_CLASSES["ProgressBar"].get(variant, "a2ui-progress-default")
        percent = clamp_percentage(value, max_val)
        return f'''<div class="a2ui-progress {variant_class}">
            <div class="a2ui-progress-fill" style="width: {percent}%"></div>
        </div>'''

    elif comp_type == "List":
        inner = "".join(render_component(c, depth + 1) for c in children)
        return f'<div class="a2ui-column">{inner}</div>'

    elif comp_type == "Image":
        alt = props.get("alt", "Image")
        src = props.get("src", "")
        alt_escaped = escape_html(alt)
        # Validate URL scheme and escape attributes to prevent XSS
        if src and src.startswith(ALLOWED_URL_SCHEMES):
            src_escaped = escape_html_attr(src)
            return f'<img class="a2ui-image" src="{src_escaped}" alt="{alt_escaped}" />'
        # Invalid or missing src - show placeholder
        return f'<div class="a2ui-image-placeholder">[{alt_escaped}]</div>'

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
        icon_key = props.get("icon", "info")
        if isinstance(icon_key, dict):
            icon_key = icon_key.get("name", "info")
        icon = ICON_MAP.get(icon_key, "ℹ️")
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
        """Serve the file browser page with version grouping."""
        from datetime import datetime
        from collections import defaultdict

        # Recursively find all JSON files
        all_files = sorted(
            self.output_path.rglob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        if not all_files:
            version_sections = '''
                <div class="empty">
                    <div class="empty-icon">📁</div>
                    <p>No JSON files found in output directory</p>
                </div>
            '''
        else:
            # Group files by version
            version_groups = defaultdict(list)

            for f in all_files:
                # Determine version from path or filename
                relative_path = f.relative_to(self.output_path)
                path_parts = relative_path.parts

                if len(path_parts) > 1:
                    # File is in a subdirectory
                    version = path_parts[0]
                else:
                    # File is in root directory - extract version from filename
                    filename = f.name
                    if 'v1_baseline' in filename or 'v1' in filename:
                        version = 'v1_baseline'
                    elif 'v2_google_gui' in filename or 'v2' in filename:
                        version = 'v2_google_gui'
                    elif 'v3_with_visual' in filename or 'v3' in filename:
                        version = 'v3_with_visual'
                    else:
                        version = 'other'

                version_groups[version].append(f)

            # Generate HTML for each version section
            sections = []
            version_display_names = {
                'v1_baseline': 'V1 Baseline',
                'v2_google_gui': 'V2 Google GUI',
                'v3_with_visual': 'V3 With Visual',
                'overlay_test': 'Overlay Test',
                'other': 'Other Files'
            }

            version_classes = {
                'v1_baseline': 'v1',
                'v2_google_gui': 'v2',
                'v3_with_visual': 'v3',
                'overlay_test': 'other',
                'other': 'other'
            }

            # Sort version groups: v1, v2, v3, others
            version_order = ['v1_baseline', 'v2_google_gui', 'v3_with_visual', 'overlay_test']
            sorted_versions = sorted(
                version_groups.keys(),
                key=lambda v: version_order.index(v) if v in version_order else 999
            )

            for version in sorted_versions:
                files = version_groups[version]
                version_name = version_display_names.get(version, version.replace('_', ' ').title())
                version_class = version_classes.get(version, 'other')

                # Generate file cards for this version
                cards = []
                for f in files:
                    stat = f.stat()
                    size_kb = stat.st_size / 1024
                    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

                    # Get relative path for preview link
                    relative_path = f.relative_to(self.output_path)
                    preview_path = str(relative_path).replace('\\', '/')

                    cards.append(f'''
                        <div class="file-card {version_class}" onclick="window.location.href='/preview?file={preview_path}'">
                            <div class="file-header">
                                <div class="file-icon">📄</div>
                                <div class="file-info">
                                    <a href="/preview?file={preview_path}" class="file-name">{f.name}</a>
                                </div>
                            </div>
                            <div class="file-meta">
                                <div class="meta-item">
                                    <span class="meta-icon">💾</span>
                                    <span>{size_kb:.1f} KB</span>
                                </div>
                                <div class="meta-item">
                                    <span class="meta-icon">🕐</span>
                                    <span>{mtime}</span>
                                </div>
                            </div>
                        </div>
                    ''')

                file_grid = "".join(cards)

                sections.append(f'''
                    <div class="version-section">
                        <div class="version-header">
                            <span class="version-badge {version_class}">{version_name}</span>
                            <span class="version-title">{len(files)} file{'s' if len(files) != 1 else ''}</span>
                            <span class="version-count">Updated {datetime.fromtimestamp(files[0].stat().st_mtime).strftime("%Y-%m-%d")}</span>
                        </div>
                        <div class="file-grid">
                            {file_grid}
                        </div>
                    </div>
                ''')

            version_sections = "".join(sections)

        html = FILE_LIST_TEMPLATE.format(version_sections=version_sections)
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
            print(f"DEBUG: Rendering component type: {comp_type}")
            
            if comp_type == "SmartGlassesSurface":
                # Unpack the stream and render internal components
                stream = comp.get("metadata", {}).get("a2ui_stream", [])
                print(f"DEBUG: Found SmartGlassesSurface with stream length: {len(stream)}")
                for msg in stream:
                    if msg.get("type") == "updateComponents":
                        internal_components = msg.get("components", [])
                        print(f"DEBUG: Found updateComponents with {len(internal_components)} components")
                        for internal_comp in internal_components:
                            component_html += render_component(internal_comp)
            elif comp_type in ["Card", "Row", "Column", "Text", "Icon", "Button"]:
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
