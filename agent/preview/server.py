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
    <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Rajdhani', sans-serif;
            background: #000;
            min-height: 100vh;
            color: #00ff41;
            padding: 20px;
        }}

        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}

        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
            color: #00ff41;
            font-weight: 700;
            text-shadow: 0 0 10px rgba(0, 255, 65, 0.4);
        }}

        .header .subtitle {{
            color: rgba(0, 255, 65, 0.7);
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
            background: rgba(34, 197, 94, 0.2);
            color: white;
            border: 1px solid #22c55e;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .controls button.secondary {{
            background: transparent;
            color: white;
            border: 1px solid #555;
            text-transform: uppercase;
            letter-spacing: 0.05em;
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
            background: url('https://images.unsplash.com/photo-1557804506-669a67965ba0?ixlib=rb-1.2.1&auto=format&fit=crop&w=1567&q=80') center/cover;
            border-radius: 4px;
            padding: 20px;
            width: 100%;
            max-width: 960px;
            min-height: 540px; /* 16:9 ratio for 960px width */
            position: relative;
            overflow-y: auto;
            overflow-x: hidden;
            box-shadow: 0 0 50px rgba(0,0,0,0.9);
        }}

        .glasses-frame::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle, transparent 40%, rgba(0,0,0,0.8) 100%);
            pointer-events: none;
            z-index: 1;
        }}

        .component-container {{
            position: relative;
            width: 100%;
            padding: 40px 20px;
            min-height: 100%;
        }}

        /* ========================================
         * A2UI Component Styles - AR Glass Theme
         * ======================================== */

        /* Card Base & Variants */
        .a2ui-card {{
            padding: 16px !important;
            color: #ffffff !important;
        }}

        .a2ui-card-glass {{
            background: rgba(0, 0, 0, 0.6) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border-left: 1px solid rgba(34, 197, 94, 0.3) !important;
            border-top: 1px solid rgba(34, 197, 94, 0.3) !important;
            border-bottom: 1px solid rgba(34, 197, 94, 0.3) !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }}

        .a2ui-card-solid {{
            background: #1e1e23 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
        }}

        .a2ui-card-outline {{
            background: transparent !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 16px !important;
            backdrop-filter: blur(8px) !important;
        }}

        .a2ui-card-alert {{
            background: rgba(239, 68, 68, 0.2) !important;
            border-left: 4px solid #ef4444 !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
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

        /* Text Base & Variants */
        .a2ui-text {{
            line-height: 1.4 !important;
            color: #00ff41 !important;
            font-family: 'Rajdhani', monospace !important;
        }}

        .a2ui-text-hero {{
            font-size: 32px !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
            letter-spacing: -0.5px !important;
            color: #ffffff !important;
        }}

        .a2ui-text-h1 {{
            font-size: 24px !important;
            font-weight: 600 !important;
            line-height: 1.3 !important;
            color: #ffffff !important;
        }}

        .a2ui-text-h2 {{
            font-size: 20px !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
            color: #ffffff !important;
        }}

        .a2ui-text-body {{
            font-size: 16px !important;
            font-weight: 400 !important;
            line-height: 1.5 !important;
        }}

        .a2ui-text-caption {{
            font-size: 12px !important;
            font-weight: 400 !important;
            color: rgba(255, 255, 255, 0.7) !important;
            line-height: 1.4 !important;
        }}

        .a2ui-text-label {{
            font-size: 11px !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            color: rgba(255, 255, 255, 0.6) !important;
        }}

        /* Icon Base & Variants */
        .a2ui-icon {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }}

        .a2ui-icon-sm {{
            width: 16px !important;
            height: 16px !important;
            font-size: 14px !important;
        }}

        .a2ui-icon-md {{
            width: 24px !important;
            height: 24px !important;
            font-size: 20px !important;
        }}

        .a2ui-icon-lg {{
            width: 32px !important;
            height: 32px !important;
            font-size: 28px !important;
        }}

        /* Button Base & Variants */
        .a2ui-button {{
            padding: 10px 20px !important;
            border: none !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            cursor: pointer !important;
            transition: all 0.2s !important;
        }}

        .a2ui-btn-primary {{
            background: #00ff41 !important;
            color: #000 !important;
            border-radius: 4px !important;
            border: 1px solid #00ff41 !important;
            box-shadow: 0 0 10px rgba(0, 255, 65, 0.3) !important;
            text-transform: uppercase !important;
            font-weight: bold !important;
        }}

        .a2ui-btn-primary:hover {{
            background: #33ff67 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 0 20px rgba(0, 255, 65, 0.5) !important;
        }}

        .a2ui-btn-secondary {{
            background: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            color: white !important;
            border-radius: 100px !important;
        }}

        .a2ui-btn-secondary:hover {{
            background: rgba(255, 255, 255, 0.2) !important;
        }}

        .a2ui-btn-ghost {{
            background: transparent !important;
            color: #60a5fa !important;
        }}

        .a2ui-btn-ghost:hover {{
            background: rgba(59, 130, 246, 0.1) !important;
        }}

        .a2ui-btn-icon {{
            background: rgba(255, 255, 255, 0.1) !important;
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }}

        .a2ui-divider {{
            height: 1px;
            background: #eee;
            width: 100%;
        }}

        /* Badge Base & Variants */
        .a2ui-badge {{
            padding: 4px 10px !important;
            border-radius: 12px !important;
            font-size: 12px !important;
            font-weight: 500 !important;
        }}

        .a2ui-badge-info {{
            background: rgba(59, 130, 246, 0.2) !important;
            color: #60a5fa !important;
        }}

        .a2ui-badge-success {{
            background: rgba(34, 197, 94, 0.2) !important;
            color: #4ade80 !important;
        }}

        .a2ui-badge-warning {{
            background: rgba(245, 158, 11, 0.2) !important;
            color: #fbbf24 !important;
        }}

        .a2ui-badge-error {{
            background: rgba(239, 68, 68, 0.2) !important;
            color: #f87171 !important;
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

# Video overlay template for real-time preview with gaze-based UI positioning
VIDEO_OVERLAY_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Overlay Preview - {sample}</title>
    <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Rajdhani', sans-serif;
            background: #0a0a0a;
            min-height: 100vh;
            color: #00ff41;
            display: flex;
            flex-direction: column;
        }}

        .header {{
            padding: 15px 20px;
            background: rgba(0, 0, 0, 0.8);
            border-bottom: 1px solid rgba(0, 255, 65, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header h1 {{
            font-size: 18px;
            font-weight: 600;
            color: #00ff41;
        }}

        .header .meta {{
            font-size: 12px;
            color: rgba(255, 255, 255, 0.6);
        }}

        .controls {{
            display: flex;
            gap: 10px;
        }}

        .controls button {{
            padding: 8px 16px;
            border: 1px solid #00ff41;
            background: rgba(0, 255, 65, 0.1);
            color: #00ff41;
            border-radius: 4px;
            cursor: pointer;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            transition: all 0.2s;
        }}

        .controls button:hover {{
            background: rgba(0, 255, 65, 0.3);
        }}

        .controls button.active {{
            background: #00ff41;
            color: #000;
        }}

        .main-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
            gap: 20px;
        }}

        .video-container {{
            position: relative;
            width: 100%;
            max-width: 1280px;
            margin: 0 auto;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 0 30px rgba(0, 255, 65, 0.2);
        }}

        .video-container video {{
            width: 100%;
            display: block;
        }}

        .ui-overlay {{
            position: absolute;
            pointer-events: none;
            transition: all 0.3s ease;
            transform: translate(-50%, -50%);
            max-width: 400px;
            z-index: 10;
        }}

        .ui-overlay.hidden {{
            opacity: 0;
        }}

        .gaze-indicator {{
            position: absolute;
            width: 20px;
            height: 20px;
            border: 2px solid #ff0000;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
            z-index: 5;
            opacity: 0.8;
        }}

        .gaze-indicator::after {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 4px;
            height: 4px;
            background: #ff0000;
            border-radius: 50%;
            transform: translate(-50%, -50%);
        }}

        .timeline {{
            background: rgba(0, 0, 0, 0.8);
            padding: 15px 20px;
            border-radius: 8px;
            border: 1px solid rgba(0, 255, 65, 0.3);
        }}

        .timeline-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 12px;
        }}

        .timeline-bar {{
            position: relative;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            cursor: pointer;
        }}

        .timeline-progress {{
            position: absolute;
            height: 100%;
            background: linear-gradient(90deg, #00ff41, #00d4ff);
            border-radius: 4px;
            width: 0%;
        }}

        .timeline-ui-window {{
            position: absolute;
            height: 100%;
            background: rgba(0, 255, 65, 0.3);
            border: 1px solid #00ff41;
            border-radius: 4px;
        }}

        .settings-panel {{
            background: rgba(0, 0, 0, 0.8);
            padding: 15px 20px;
            border-radius: 8px;
            border: 1px solid rgba(0, 255, 65, 0.3);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}

        .setting-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}

        .setting-group label {{
            font-size: 11px;
            text-transform: uppercase;
            color: rgba(255, 255, 255, 0.6);
        }}

        .setting-group input, .setting-group select {{
            padding: 8px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(0, 255, 65, 0.3);
            border-radius: 4px;
            color: #fff;
            font-family: 'Rajdhani', sans-serif;
        }}

        .a2ui-card {{
            padding: 16px !important;
            color: #ffffff !important;
        }}

        .a2ui-card-glass {{
            background: rgba(0, 0, 0, 0.7) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(34, 197, 94, 0.4) !important;
            border-radius: 12px !important;
            box-shadow: 0 0 20px rgba(0, 255, 65, 0.2) !important;
        }}

        .a2ui-row {{ display: flex; align-items: center; gap: 12px; }}
        .a2ui-column {{ display: flex; flex-direction: column; gap: 8px; }}
        .a2ui-text {{ line-height: 1.4 !important; color: #00ff41 !important; font-family: 'Rajdhani', monospace !important; }}
        .a2ui-text-hero {{ font-size: 32px !important; font-weight: 700 !important; color: #ffffff !important; }}
        .a2ui-text-h1 {{ font-size: 24px !important; font-weight: 600 !important; color: #ffffff !important; }}
        .a2ui-text-h2 {{ font-size: 20px !important; font-weight: 600 !important; color: #ffffff !important; }}
        .a2ui-text-body {{ font-size: 16px !important; font-weight: 400 !important; }}
        .a2ui-text-caption {{ font-size: 12px !important; color: rgba(255, 255, 255, 0.7) !important; }}
        .a2ui-text-label {{ font-size: 11px !important; text-transform: uppercase !important; color: rgba(255, 255, 255, 0.6) !important; }}
        .a2ui-icon {{ display: flex !important; align-items: center !important; justify-content: center !important; }}
        .a2ui-icon-sm {{ width: 16px !important; height: 16px !important; font-size: 14px !important; }}
        .a2ui-icon-md {{ width: 24px !important; height: 24px !important; font-size: 20px !important; }}
        .a2ui-icon-lg {{ width: 32px !important; height: 32px !important; font-size: 28px !important; }}
        .a2ui-button {{ padding: 10px 20px !important; border: none !important; font-size: 14px !important; cursor: pointer !important; }}
        .a2ui-btn-primary {{ background: #00ff41 !important; color: #000 !important; border-radius: 4px !important; font-weight: bold !important; }}
        .a2ui-badge {{ padding: 4px 10px !important; border-radius: 12px !important; font-size: 12px !important; }}
        .a2ui-badge-info {{ background: rgba(59, 130, 246, 0.2) !important; color: #60a5fa !important; }}
        .a2ui-badge-success {{ background: rgba(34, 197, 94, 0.2) !important; color: #4ade80 !important; }}

        .export-panel {{
            background: rgba(0, 0, 0, 0.9);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #00ff41;
            margin-top: 10px;
        }}

        .export-panel h3 {{
            margin-bottom: 15px;
            color: #00ff41;
        }}

        .export-status {{
            padding: 10px;
            background: rgba(0, 255, 65, 0.1);
            border-radius: 4px;
            font-size: 14px;
            margin-top: 10px;
        }}

        .json-view {{
            background: #1e1e1e;
            border-radius: 8px;
            padding: 15px;
            max-height: 200px;
            overflow: auto;
            font-size: 12px;
            display: none;
        }}

        .json-view.visible {{
            display: block;
        }}

        .json-view pre {{
            font-family: 'Monaco', 'Menlo', monospace;
            color: #d4d4d4;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🎬 Video Overlay Preview</h1>
            <span class="meta">{sample} | Strategy: {strategy}</span>
        </div>
        <div class="controls">
            <button onclick="location.href='/'">📁 Back</button>
            <button id="btn-gaze" class="active" onclick="toggleGaze()">👁 Gaze</button>
            <button id="btn-json" onclick="toggleJson()">📋 JSON</button>
            <button id="btn-export" onclick="toggleExport()">📹 Export</button>
        </div>
    </div>

    <div class="main-container">
        <div class="video-container" id="video-container">
            <video id="video" src="{video_url}" preload="auto"></video>
            <div class="gaze-indicator" id="gaze-indicator" style="display: none;"></div>
            <div class="ui-overlay hidden" id="ui-overlay">
                {component_html}
            </div>
        </div>

        <div class="timeline">
            <div class="timeline-header">
                <span id="current-time">00:00.000</span>
                <span>UI Window: <span id="ui-window-info">Last 2.0s</span></span>
                <span id="total-time">00:00.000</span>
            </div>
            <div class="timeline-bar" id="timeline-bar" onclick="seekVideo(event)">
                <div class="timeline-progress" id="timeline-progress"></div>
                <div class="timeline-ui-window" id="timeline-ui-window"></div>
            </div>
        </div>

        <div class="settings-panel">
            <div class="setting-group">
                <label>UI Display Duration (seconds)</label>
                <input type="number" id="ui-duration" value="2.0" min="0.5" max="10" step="0.5" onchange="updateUIWindow()">
            </div>
            <div class="setting-group">
                <label>UI Position Mode</label>
                <select id="position-mode" onchange="updatePositionMode()">
                    <option value="gaze">Follow Gaze</option>
                    <option value="center">Center</option>
                    <option value="fixed">Fixed Position</option>
                </select>
            </div>
            <div class="setting-group">
                <label>Gaze Fallback</label>
                <select id="gaze-fallback">
                    <option value="last_valid">Last Valid</option>
                    <option value="center">Center</option>
                    <option value="interpolate">Interpolate</option>
                </select>
            </div>
            <div class="setting-group">
                <label>Playback</label>
                <button onclick="playPause()" style="width: 100%;">▶ Play / ⏸ Pause</button>
            </div>
        </div>

        <div class="export-panel" id="export-panel" style="display: none;">
            <h3>📹 Export Video with Overlay</h3>
            <p style="color: rgba(255,255,255,0.7); margin-bottom: 15px;">
                Capture the current preview as a video file with UI overlay.
            </p>
            <div class="controls">
                <button onclick="startRecording()">🔴 Start Recording</button>
                <button onclick="stopRecording()" id="stop-btn" disabled>⏹ Stop & Save</button>
                <button onclick="captureScreenshot()">📷 Screenshot</button>
            </div>
            <div class="export-status" id="export-status">Ready to record</div>
        </div>

        <div class="json-view" id="json-view">
            <pre>{json_content}</pre>
        </div>
    </div>

    <script>
        // State
        let gazeData = [];
        let rawData = null;
        let mediaRecorder = null;
        let recordedChunks = [];
        let showGaze = true;

        const video = document.getElementById('video');
        const uiOverlay = document.getElementById('ui-overlay');
        const gazeIndicator = document.getElementById('gaze-indicator');
        const timelineProgress = document.getElementById('timeline-progress');
        const timelineUIWindow = document.getElementById('timeline-ui-window');

        // Load gaze data and rawdata
        async function loadData() {{
            const sample = '{sample}';

            // Load gaze data
            try {{
                const gazeResp = await fetch(`/api/gaze?sample=${{sample}}`);
                const gazeJson = await gazeResp.json();
                gazeData = gazeJson.data || [];
                console.log(`Loaded ${{gazeData.length}} gaze points`);
            }} catch (e) {{
                console.warn('Failed to load gaze data:', e);
            }}

            // Load rawdata for time interval
            try {{
                const rawResp = await fetch(`/api/rawdata?sample=${{sample}}`);
                rawData = await rawResp.json();
                console.log('Loaded rawdata:', rawData);
            }} catch (e) {{
                console.warn('Failed to load rawdata:', e);
            }}
        }}

        // Format time as MM:SS.mmm
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${{mins.toString().padStart(2, '0')}}:${{secs.toFixed(3).padStart(6, '0')}}`;
        }}

        // Update UI based on current video time
        function updateUI() {{
            const currentTime = video.currentTime;
            const duration = video.duration || 1;
            const uiDuration = parseFloat(document.getElementById('ui-duration').value);

            // Update timeline
            const progress = (currentTime / duration) * 100;
            timelineProgress.style.width = `${{progress}}%`;
            document.getElementById('current-time').textContent = formatTime(currentTime);
            document.getElementById('total-time').textContent = formatTime(duration);

            // Calculate UI display window (last N seconds)
            const uiStartTime = duration - uiDuration;
            const uiEndTime = duration;

            // Update UI window indicator on timeline
            const windowStart = (uiStartTime / duration) * 100;
            const windowWidth = (uiDuration / duration) * 100;
            timelineUIWindow.style.left = `${{windowStart}}%`;
            timelineUIWindow.style.width = `${{windowWidth}}%`;

            // Show/hide UI overlay based on current time
            if (currentTime >= uiStartTime && currentTime <= uiEndTime) {{
                uiOverlay.classList.remove('hidden');
            }} else {{
                uiOverlay.classList.add('hidden');
            }}

            // Update gaze indicator and UI position
            updateGazePosition(currentTime);
        }}

        // Find gaze position at given time
        function getGazeAt(time) {{
            if (gazeData.length === 0) return null;

            // Binary search for closest time
            let low = 0, high = gazeData.length - 1;
            while (low < high) {{
                const mid = Math.floor((low + high) / 2);
                if (gazeData[mid].time < time) {{
                    low = mid + 1;
                }} else {{
                    high = mid;
                }}
            }}

            const point = gazeData[low];
            if (point && point.x !== null && point.y !== null) {{
                return {{ x: point.x, y: point.y }};
            }}

            // Fallback: find last valid point
            const fallback = document.getElementById('gaze-fallback').value;
            if (fallback === 'last_valid') {{
                for (let i = low - 1; i >= 0; i--) {{
                    if (gazeData[i].x !== null && gazeData[i].y !== null) {{
                        return {{ x: gazeData[i].x, y: gazeData[i].y }};
                    }}
                }}
            }}

            // Default center
            return {{ x: video.videoWidth / 2, y: video.videoHeight / 2 }};
        }}

        // Update gaze indicator and UI position
        function updateGazePosition(currentTime) {{
            const container = document.getElementById('video-container');
            const rect = container.getBoundingClientRect();
            const videoRect = video.getBoundingClientRect();

            // Get gaze position
            const gaze = getGazeAt(currentTime);
            if (!gaze) return;

            // Scale gaze coordinates to display size
            const scaleX = videoRect.width / video.videoWidth;
            const scaleY = videoRect.height / video.videoHeight;
            const displayX = gaze.x * scaleX;
            const displayY = gaze.y * scaleY;

            // Update gaze indicator
            if (showGaze) {{
                gazeIndicator.style.display = 'block';
                gazeIndicator.style.left = `${{displayX}}px`;
                gazeIndicator.style.top = `${{displayY}}px`;
            }} else {{
                gazeIndicator.style.display = 'none';
            }}

            // Update UI position based on mode
            const mode = document.getElementById('position-mode').value;
            if (mode === 'gaze') {{
                uiOverlay.style.left = `${{displayX}}px`;
                uiOverlay.style.top = `${{displayY}}px`;
            }} else if (mode === 'center') {{
                uiOverlay.style.left = '50%';
                uiOverlay.style.top = '50%';
            }} else {{
                // Fixed position
                uiOverlay.style.left = '50%';
                uiOverlay.style.top = '70%';
            }}
        }}

        // Timeline click to seek
        function seekVideo(event) {{
            const bar = document.getElementById('timeline-bar');
            const rect = bar.getBoundingClientRect();
            const ratio = (event.clientX - rect.left) / rect.width;
            video.currentTime = ratio * video.duration;
        }}

        // Play/pause toggle
        function playPause() {{
            if (video.paused) {{
                video.play();
            }} else {{
                video.pause();
            }}
        }}

        // Toggle gaze indicator
        function toggleGaze() {{
            showGaze = !showGaze;
            document.getElementById('btn-gaze').classList.toggle('active', showGaze);
        }}

        // Toggle JSON view
        function toggleJson() {{
            document.getElementById('json-view').classList.toggle('visible');
            document.getElementById('btn-json').classList.toggle('active');
        }}

        // Toggle export panel
        function toggleExport() {{
            const panel = document.getElementById('export-panel');
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
            document.getElementById('btn-export').classList.toggle('active');
        }}

        // Update UI window display
        function updateUIWindow() {{
            const duration = parseFloat(document.getElementById('ui-duration').value);
            document.getElementById('ui-window-info').textContent = `Last ${{duration.toFixed(1)}}s`;
            updateUI();
        }}

        // Update position mode
        function updatePositionMode() {{
            updateUI();
        }}

        // Recording functionality
        async function startRecording() {{
            try {{
                const container = document.getElementById('video-container');
                const stream = await navigator.mediaDevices.getDisplayMedia({{
                    video: {{ cursor: 'never' }},
                    audio: false
                }});

                recordedChunks = [];
                mediaRecorder = new MediaRecorder(stream, {{ mimeType: 'video/webm' }});

                mediaRecorder.ondataavailable = (e) => {{
                    if (e.data.size > 0) recordedChunks.push(e.data);
                }};

                mediaRecorder.onstop = () => {{
                    const blob = new Blob(recordedChunks, {{ type: 'video/webm' }});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `overlay_{sample}_{strategy}.webm`;
                    a.click();
                    URL.revokeObjectURL(url);
                    document.getElementById('export-status').textContent = 'Recording saved!';
                }};

                mediaRecorder.start();
                document.getElementById('stop-btn').disabled = false;
                document.getElementById('export-status').textContent = '🔴 Recording...';

                // Auto-play video
                video.currentTime = 0;
                video.play();
            }} catch (e) {{
                document.getElementById('export-status').textContent = `Error: ${{e.message}}`;
            }}
        }}

        function stopRecording() {{
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {{
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
                document.getElementById('stop-btn').disabled = true;
            }}
        }}

        // Screenshot functionality
        function captureScreenshot() {{
            const container = document.getElementById('video-container');

            // Use html2canvas if available, otherwise use canvas approach
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');

            // Draw video frame
            ctx.drawImage(video, 0, 0);

            // Download
            canvas.toBlob((blob) => {{
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `screenshot_{sample}_{strategy}.png`;
                a.click();
                URL.revokeObjectURL(url);
                document.getElementById('export-status').textContent = 'Screenshot saved!';
            }});
        }}

        // Event listeners
        video.addEventListener('timeupdate', updateUI);
        video.addEventListener('loadedmetadata', () => {{
            updateUIWindow();
            updateUI();
        }});

        // Initialize
        loadData();
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

        /* Task badge for hierarchical structure */
        .version-badge.task {{
            background: rgba(255, 215, 0, 0.15);
            color: #ffd700;
            border: 1px solid rgba(255, 215, 0, 0.3);
        }}

        /* Strategy badges for sample cards */
        .strategy-badges {{
            display: flex;
            gap: 8px;
            margin-top: 8px;
            flex-wrap: wrap;
        }}

        .strategy-badge {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.2s;
        }}

        .strategy-badge.v1 {{
            background: rgba(0, 212, 255, 0.2);
            color: var(--accent-v1);
            border: 1px solid rgba(0, 212, 255, 0.4);
        }}

        .strategy-badge.v2 {{
            background: rgba(0, 255, 136, 0.2);
            color: var(--accent-v2);
            border: 1px solid rgba(0, 255, 136, 0.4);
        }}

        .strategy-badge.v3 {{
            background: rgba(189, 0, 255, 0.2);
            color: var(--accent-v3);
            border: 1px solid rgba(189, 0, 255, 0.4);
        }}

        .strategy-badge.other {{
            background: rgba(255, 107, 53, 0.2);
            color: var(--accent-other);
            border: 1px solid rgba(255, 107, 53, 0.4);
        }}

        .strategy-badge.video {{
            background: rgba(255, 0, 128, 0.2);
            color: #ff0080;
            border: 1px solid rgba(255, 0, 128, 0.4);
        }}

        .strategy-badge:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}

        /* Sample card styling */
        .sample-card {{
            cursor: default;
        }}

        .sample-card::before {{
            background: linear-gradient(180deg, var(--accent-v1), var(--accent-v2), var(--accent-v3)) !important;
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
        # Support both "text" and "content" in props, plus "content" at root level
        text = props.get("text", "") or props.get("content", "") or component.get("content", "")
        variant = props.get("variant", "body")
        variant_class = VARIANT_CLASSES["Text"].get(variant, "a2ui-text-body")
        text_escaped = escape_html(text)
        return f'<span class="a2ui-text {variant_class}">{text_escaped}</span>'

    elif comp_type == "Icon":
        # Extract icon properties - support both "size" and "variant" for size
        name = props.get("name", "info")
        size = props.get("size", "") or props.get("variant", "medium")
        size_class = VARIANT_CLASSES["Icon"].get(size, "a2ui-icon-md")
        # Ensure name is a string for ICON_MAP
        if isinstance(name, dict):
            name = name.get("name", "info")
        icon = ICON_MAP.get(name, "❓")
        return f'<span class="a2ui-icon {size_class}">{icon}</span>'

    elif comp_type == "Button":
        # Support both "label" and "content" for button text
        label = props.get("label", "") or props.get("content", "Button")
        variant = props.get("variant", "primary")
        variant_class = VARIANT_CLASSES["Button"].get(variant, "a2ui-btn-primary")
        label_escaped = escape_html(label)
        return f'<button class="a2ui-button {variant_class}">{label_escaped}</button>'

    elif comp_type == "Divider":
        return '<div class="a2ui-divider"></div>'

    elif comp_type == "Badge":
        # Support both "text" and "content" for badge text
        text = props.get("text", "") or props.get("content", "")
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
    example_path: Path = Path(__file__).parent.parent / "example"

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
        elif path == "/video-overlay":
            # New video overlay preview endpoint
            sample = query.get("sample", [None])[0]
            # Support both 'file' (new: strategy_model) and 'strategy' (legacy) parameters
            file_param = query.get("file", [None])[0]
            strategy = query.get("strategy", ["v2_google_gui"])[0]
            if sample:
                self.serve_video_overlay(sample, file_param or strategy)
            else:
                self.send_error(400, "Missing sample parameter")
        elif path == "/api/gaze":
            # API endpoint for gaze data
            sample = query.get("sample", [None])[0]
            if sample:
                self.serve_gaze_data(sample)
            else:
                self.send_error(400, "Missing sample parameter")
        elif path == "/api/rawdata":
            # API endpoint for sample metadata
            sample = query.get("sample", [None])[0]
            if sample:
                self.serve_rawdata(sample)
            else:
                self.send_error(400, "Missing sample parameter")
        elif path.startswith("/video/"):
            # Serve video files from example directory
            self.serve_video_file(path[7:])  # Remove "/video/" prefix
        elif path.startswith("/static/"):
            super().do_GET()
        else:
            self.send_error(404, "Not Found")

    def serve_file_list(self):
        """Serve the file browser page with hierarchical Task/Participant/Sample grouping."""
        from datetime import datetime
        from collections import defaultdict

        # Supported strategy prefixes for hierarchical structure
        strategy_prefixes = {"v1_baseline", "v2_google_gui", "v3_with_visual", "v2_smart_glasses"}

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
            # Separate hierarchical files (Task/Participant/Sample/strategy.json) from legacy flat files
            hierarchical_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            legacy_files = defaultdict(list)

            for f in all_files:
                relative_path = f.relative_to(self.output_path)
                path_parts = relative_path.parts

                # Check if this is a hierarchical structure: Task2.x/Participant/sample_xxx/{strategy}_{model}.json
                # New format: v2_google_gui_azure_gpt-4o.json
                # Old format: v2_google_gui.json
                if len(path_parts) >= 4 and path_parts[0].startswith('Task'):
                    # Check if filename starts with a known strategy prefix
                    filename_stem = f.stem
                    matched_strategy = None
                    model_name = None

                    for prefix in strategy_prefixes:
                        if filename_stem.startswith(prefix):
                            matched_strategy = prefix
                            # Extract model name if present (after strategy prefix)
                            remainder = filename_stem[len(prefix):]
                            if remainder.startswith('_'):
                                model_name = remainder[1:]  # Remove leading underscore
                            break

                    if matched_strategy:
                        task = path_parts[0]
                        participant = path_parts[1]
                        sample = path_parts[2]

                        hierarchical_files[task][participant][sample].append({
                            "strategy": matched_strategy,
                            "model": model_name,
                            "filename": filename_stem,  # Full filename without extension
                            "path": str(relative_path).replace('\\', '/'),
                            "file": f,
                        })
                    else:
                        # Unknown format, treat as legacy
                        legacy_files['other'].append(f)
                else:
                    # Legacy flat file - group by version extracted from filename
                    filename = f.name
                    if 'v1_baseline' in filename or 'v1' in filename:
                        version = 'v1_baseline'
                    elif 'v2_google_gui' in filename or 'v2' in filename:
                        version = 'v2_google_gui'
                    elif 'v3_with_visual' in filename or 'v3' in filename:
                        version = 'v3_with_visual'
                    else:
                        version = 'other'
                    legacy_files[version].append(f)

            sections = []

            # Strategy display configuration
            strategy_display = {
                'v1_baseline': ('V1', 'v1'),
                'v2_google_gui': ('V2', 'v2'),
                'v3_with_visual': ('V3', 'v3'),
                'v2_smart_glasses': ('SG', 'v2'),
            }

            # Generate sections for hierarchical files (Task → Participant → Sample)
            for task in sorted(hierarchical_files.keys()):
                participants = hierarchical_files[task]

                for participant in sorted(participants.keys()):
                    samples = participants[participant]

                    # Generate sample cards for this participant
                    cards = []
                    for sample in sorted(samples.keys()):
                        strategies = samples[sample]

                        # Build strategy badges with model info
                        strategy_badges = []
                        for s in sorted(strategies, key=lambda x: (x['strategy'], x.get('model') or '')):
                            base_strategy = s['strategy']
                            model = s.get('model')
                            filename = s['filename']

                            # Determine display name and badge class
                            display_info = strategy_display.get(base_strategy, (base_strategy[:2].upper(), 'other'))
                            base_name, badge_class = display_info

                            # Add model indicator if present
                            if model:
                                # Extract short model name (e.g., "azure_gpt-4o" -> "GPT4o")
                                model_short = model.split('_')[-1].replace('-', '').replace('.', '')[:6].upper()
                                display_name = f"{base_name}:{model_short}"
                                title = f"{base_strategy} + {model}"
                            else:
                                display_name = base_name
                                title = base_strategy

                            strategy_badges.append(
                                f'<a href="/preview?file={s["path"]}" class="strategy-badge {badge_class}" title="{title}">{display_name}</a>'
                            )

                        # Get file stats from first strategy file
                        first_file = strategies[0]['file']
                        first_filename = strategies[0]['filename']
                        stat = first_file.stat()
                        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

                        cards.append(f'''
                            <div class="file-card sample-card">
                                <div class="file-header">
                                    <div class="file-icon">📁</div>
                                    <div class="file-info">
                                        <span class="file-name">{sample}</span>
                                        <div class="strategy-badges">
                                            {''.join(strategy_badges)}
                                            <a href="/video-overlay?sample={task}/{participant}/{sample}&file={first_filename}" class="strategy-badge video" title="Video Overlay Preview">🎬</a>
                                        </div>
                                    </div>
                                </div>
                                <div class="file-meta">
                                    <div class="meta-item">
                                        <span class="meta-icon">📊</span>
                                        <span>{len(strategies)} variants</span>
                                    </div>
                                    <div class="meta-item">
                                        <span class="meta-icon">🕐</span>
                                        <span>{mtime}</span>
                                    </div>
                                </div>
                            </div>
                        ''')

                    file_grid = "".join(cards)
                    sample_count = len(samples)

                    sections.append(f'''
                        <div class="version-section">
                            <div class="version-header">
                                <span class="version-badge task">{task}</span>
                                <span class="version-title">{participant}</span>
                                <span class="version-count">{sample_count} sample{'s' if sample_count != 1 else ''}</span>
                            </div>
                            <div class="file-grid">
                                {file_grid}
                            </div>
                        </div>
                    ''')

            # Generate sections for legacy flat files
            version_display_names = {
                'v1_baseline': 'V1 Baseline (Legacy)',
                'v2_google_gui': 'V2 Google GUI (Legacy)',
                'v3_with_visual': 'V3 With Visual (Legacy)',
                'other': 'Other Files'
            }

            version_classes = {
                'v1_baseline': 'v1',
                'v2_google_gui': 'v2',
                'v3_with_visual': 'v3',
                'other': 'other'
            }

            version_order = ['v1_baseline', 'v2_google_gui', 'v3_with_visual', 'other']
            sorted_versions = sorted(
                legacy_files.keys(),
                key=lambda v: version_order.index(v) if v in version_order else 999
            )

            for version in sorted_versions:
                files = legacy_files[version]
                if not files:
                    continue

                version_name = version_display_names.get(version, version.replace('_', ' ').title())
                version_class = version_classes.get(version, 'other')

                cards = []
                for f in files:
                    stat = f.stat()
                    size_kb = stat.st_size / 1024
                    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

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

    def send_json(self, data: dict):
        """Send JSON response."""
        json_str = json.dumps(data, ensure_ascii=False)
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(json_str.encode("utf-8")))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json_str.encode("utf-8"))

    def serve_video_file(self, video_path: str):
        """Serve video files from example directory."""
        # video_path format: Task2.1/P1_YuePan/sample_027/video/clip.mp4
        full_path = self.example_path / video_path

        if not full_path.exists():
            self.send_error(404, f"Video not found: {video_path}")
            return

        # Get file size for Content-Length
        file_size = full_path.stat().st_size

        # Handle range requests for video seeking
        range_header = self.headers.get('Range')
        if range_header:
            # Parse range header
            range_match = range_header.replace('bytes=', '').split('-')
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1

            self.send_response(206)  # Partial Content
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", end - start + 1)
        else:
            start = 0
            end = file_size - 1
            self.send_response(200)
            self.send_header("Content-Length", file_size)

        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Send file content
        with open(full_path, 'rb') as f:
            f.seek(start)
            remaining = end - start + 1
            chunk_size = 64 * 1024  # 64KB chunks
            while remaining > 0:
                chunk = f.read(min(chunk_size, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def serve_gaze_data(self, sample: str):
        """Serve gaze data as JSON."""
        import csv

        # sample format: Task2.1/P1_YuePan/sample_027
        gaze_path = self.example_path / sample / "signals" / "gaze.csv"

        if not gaze_path.exists():
            self.send_json({"error": "Gaze data not found", "data": []})
            return

        gaze_data = []
        with open(gaze_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    time_s = float(row.get('time_s', 0))
                    x_str = row.get('gaze x [px]', '').strip()
                    y_str = row.get('gaze y [px]', '').strip()

                    gaze_data.append({
                        'time': time_s,
                        'x': float(x_str) if x_str else None,
                        'y': float(y_str) if y_str else None,
                    })
                except (ValueError, KeyError):
                    continue

        self.send_json({"data": gaze_data})

    def serve_rawdata(self, sample: str):
        """Serve sample rawdata.json."""
        rawdata_path = self.example_path / sample / "rawdata.json"

        if not rawdata_path.exists():
            self.send_json({"error": "rawdata.json not found"})
            return

        with open(rawdata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.send_json(data)

    def serve_video_overlay(self, sample: str, strategy: str):
        """Serve the video overlay preview page."""
        # Load UI JSON
        ui_json_path = self.output_path / sample / f"{strategy}.json"

        if not ui_json_path.exists():
            self.send_error(404, f"UI JSON not found: {ui_json_path}")
            return

        try:
            with open(ui_json_path, 'r', encoding='utf-8') as f:
                ui_data = json.load(f)
        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {e}")
            return

        # Render UI component
        if isinstance(ui_data, list):
            components = ui_data[:1]
        else:
            components = [ui_data]

        component_html = ""
        for comp in components:
            comp_type = comp.get("type", "")
            if comp_type == "SmartGlassesSurface":
                stream = comp.get("metadata", {}).get("a2ui_stream", [])
                for msg in stream:
                    if msg.get("type") == "updateComponents":
                        for internal_comp in msg.get("components", []):
                            component_html += render_component(internal_comp)
            elif comp_type in ["Card", "Row", "Column", "Text", "Icon", "Button"]:
                component_html += render_component(comp)
            else:
                component_html += render_legacy_component(comp)

        # Build video URL
        video_url = f"/video/{sample}/video/clip.mp4"

        html = VIDEO_OVERLAY_TEMPLATE.format(
            sample=sample,
            strategy=strategy,
            video_url=video_url,
            component_html=component_html,
            json_content=json.dumps(ui_data, indent=2, ensure_ascii=False),
        )

        self.send_html(html)

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
