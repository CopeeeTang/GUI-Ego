# A2UI Preview

A lightweight web-based viewer for A2UI component JSON outputs.

## Quick Start

```bash
# Navigate to project root
cd /home/v-tangxin/GUI

# Start server
python -m agent.preview.server
```

Server runs at: **http://localhost:8080**

## Stop Server

Press `Ctrl+C` in the terminal

## Features

- 📁 **Version Groups** - Auto-organized by v1/v2/v3
- 🔍 **Live Search** - Filter files by name or version
- 🎨 **Visual Preview** - Rendered A2UI components
- 📊 **File Stats** - Size, timestamp, metadata

## Options

```bash
# Custom port
python -m agent.preview.server --port 8000

# Custom output directory
python -m agent.preview.server --output-path /path/to/output

# All options
python -m agent.preview.server --help
```

## File Structure

```
agent/output/
├── v1_baseline/         → V1 files
├── v2_google_gui/       → V2 files
├── v3_with_visual/      → V3 files
└── *.json              → Auto-detected version
```

All JSON files are automatically discovered and grouped by version.
