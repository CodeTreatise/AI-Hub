# AI Knowledge Graph Viewer

A D3.js-based interactive visualization tool for exploring the AI Landscape and Generative AI Learning Path.

## 📂 Project Structure

```
graph-viewer/
├── assets/             # Static assets
│   ├── css/            # Stylesheets (main.css, viewer.css)
│   └── js/             # Frontend scripts (D3.js logic)
├── data/               # Data files
│   ├── graphs/         # Processed JSON graph files used by the viewer
│   ├── source/         # Source of Truth (Markdown files)
│   │   ├── AI-Landscape.md
│   │   └── GENAI_LEARNING_PATH.md
│   ├── landscape.json  # Main dataset generated from AI-Landscape.md
│   └── metadata.json   # Enriched metadata (titles, descriptions)
├── pages/              # HTML Views
│   ├── viewer.html     # Main graph visualization page
│   └── ...
├── tools/              # Python utilities for data processing
│   ├── parse_landscape.py
│   ├── fetch_metadata.py
│   ├── check_urls.py
│   └── validate_landscape.py
└── index.html          # Entry point
```

## 🔄 Data Pipeline

The data flow works as follows:

1.  **Source**: The content is authored in Markdown files in `data/source/`.
    *   `AI-Landscape.md`: Contains Mermaid diagrams defining the nodes and edges.
2.  **Extraction**: `tools/parse_landscape.py` reads the Markdown and converts it into a structured JSON file (`data/landscape.json`).
3.  **Enrichment**: `tools/fetch_metadata.py` reads the JSON, visits the URLs for each node, and fetches metadata (titles, descriptions) to create `data/metadata.json`.
4.  **Validation**: `tools/check_urls.py` and `tools/validate_landscape.py` ensure data integrity.
5.  **Visualization**: The frontend (`assets/js/viewer.js`) loads the JSON data to render the interactive graph.

## 🛠️ Tools Usage

All tools should be run from the `tools/` directory.

### 1. Update Data from Markdown
If you edit `AI-Landscape.md`, run this to update the JSON:
```bash
cd tools
python3 parse_landscape.py
```

### 2. Fetch Metadata
To update link previews and descriptions:
```bash
cd tools
python3 fetch_metadata.py
```

### 3. Validate Links
To check for broken URLs:
```bash
cd tools
python3 check_urls.py
```
