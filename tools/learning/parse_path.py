#!/usr/bin/env python3
"""
Parse GENAI_LEARNING_PATH.md into a structured JSON graph.
Extracts narrative content and saves it as local HTML guides.
"""

import re
import json
import os
import hashlib
from pathlib import Path
from markdown_it import MarkdownIt

# Configuration
SOURCE_FILE = "../../data/source/GENAI_LEARNING_PATH.md"
OUTPUT_JSON = "../../data/graphs/genai-path.json"
CURATED_DIR = "../../data/curated"
CSS_PATH = "../../assets/css/main.css"

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="../../assets/css/main.css">
    <style>
        body {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background-color: #0f0f1a;
        }}
        .reader-container {{
            background: #1a1a2e;
            padding: 3rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .meta-header {{
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #30363d;
        }}
        
        /* Markdown Content Styles */
        h1, h2, h3 {{ color: #e0e0e0; margin-top: 1.5rem; }}
        p {{ line-height: 1.6; color: #c9d1d9; margin-bottom: 1rem; }}
        ul, ol {{ color: #c9d1d9; margin-bottom: 1rem; padding-left: 2rem; }}
        li {{ margin-bottom: 0.5rem; }}
        pre {{ background: #0d1117; padding: 1rem; border-radius: 6px; overflow-x: auto; margin: 1rem 0; }}
        code {{ font-family: 'Fira Code', monospace; color: #e0e0e0; background: rgba(110, 118, 129, 0.4); padding: 0.2em 0.4em; border-radius: 6px; }}
        pre code {{ background: transparent; padding: 0; }}
        a {{ color: #58a6ff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        blockquote {{ border-left: 4px solid #58a6ff; padding-left: 1rem; color: #8b949e; margin: 1rem 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #30363d; padding: 0.5rem; text-align: left; color: #c9d1d9; }}
        th {{ background: #161b22; }}
    </style>
</head>
<body>
    <div class="reader-container">
        <div class="meta-header">
            <h1>{title}</h1>
            <p>Type: {type}</p>
        </div>
        <div class="content">
            {content}
        </div>
    </div>
</body>
</html>
"""

def save_guide(node_id, title, type_label, markdown_content):
    if not markdown_content.strip():
        return None
        
    md = MarkdownIt()
    html_content = md.render(markdown_content)
    
    full_html = HTML_TEMPLATE.format(
        title=title,
        type=type_label,
        content=html_content
    )
    
    filename = f"{node_id}.html"
    filepath = os.path.join(CURATED_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_html)
        
    return f"data/curated/{filename}"

def parse_learning_path(md_path):
    text = Path(md_path).read_text(encoding='utf-8')
    
    nodes = []
    edges = []
    
    # Root node
    root_id = "root"
    nodes.append({
        "id": root_id,
        "label": "GenAI Learning Path",
        "type": "root",
        "level": 0
    })
    
    # State tracking
    current_phase = None
    current_topic = None
    current_node = None # The node currently accumulating text
    buffer = [] # Accumulates markdown lines for the current node
    
    def flush_buffer():
        nonlocal buffer, current_node
        if current_node and buffer:
            content = "\n".join(buffer)
            local_path = save_guide(current_node['id'], current_node['label'], current_node['type'], content)
            if local_path:
                current_node['local_path'] = local_path
            buffer = []

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 1. Phase Header: ## Phase 1: ...
        phase_match = re.match(r'^##\s+(Phase\s+\d+[:\s].+)$', line)
        if phase_match:
            flush_buffer()
            
            label = phase_match.group(1)
            phase_id = hashlib.md5(label.encode()).hexdigest()[:8]
            
            node = {
                "id": phase_id,
                "label": label,
                "type": "phase",
                "level": 1
            }
            nodes.append(node)
            edges.append({"source": root_id, "target": phase_id})
            
            current_phase = phase_id
            current_topic = None
            current_node = node
            buffer = [] # Start new buffer for Phase description
            i += 1
            continue

        # 2. Recipe Header: ## Week X Recipe
        recipe_match = re.match(r'^##\s+(Week\s+\d+\s+Recipe.*)$', line)
        if recipe_match:
            flush_buffer()
            
            label = recipe_match.group(1)
            recipe_id = hashlib.md5(label.encode()).hexdigest()[:8]
            
            node = {
                "id": recipe_id,
                "label": "📝 " + label, # Add icon to distinguish
                "type": "recipe",
                "level": 2
            }
            nodes.append(node)
            # Connect recipe to the current phase (or root if no phase)
            parent = current_phase if current_phase else root_id
            edges.append({"source": parent, "target": recipe_id})
            
            current_topic = None # Recipe is not a topic
            current_node = node
            buffer = []
            i += 1
            continue
            
        # 3. Topic Header: ### 1.1 Topic Name
        topic_match = re.match(r'^###\s+(\d+\.\d+\s+.+)$', line)
        if topic_match and current_phase:
            flush_buffer()
            
            label = topic_match.group(1)
            topic_id = hashlib.md5(label.encode()).hexdigest()[:8]
            
            node = {
                "id": topic_id,
                "label": label,
                "type": "topic",
                "level": 2
            }
            nodes.append(node)
            edges.append({"source": current_phase, "target": topic_id})
            
            current_topic = topic_id
            current_node = node
            buffer = []
            i += 1
            continue
            
        # 4. Resources Section
        if line == "**Resources:**" and current_topic:
            # Don't flush buffer yet, resources might be part of topic text?
            # Actually, let's flush topic text before resources, 
            # OR keep resources in the topic text too.
            # Let's keep them in the text so the guide is complete.
            buffer.append(line)
            
            i += 1
            while i < len(lines):
                res_line = lines[i].strip()
                
                # Stop if we hit a new header or empty line followed by header
                if res_line.startswith('#') or res_line.startswith('---'):
                    break
                
                buffer.append(lines[i]) # Add to topic guide
                
                # Extract Resource Node
                res_match = re.match(r'^\d+\.\s+\*\*(.+?)\*\*', res_line)
                if res_match:
                    title = res_match.group(1)
                    url = None
                    
                    # Look ahead for URL
                    if i + 1 < len(lines):
                        next_line = lines[i+1].strip()
                        if next_line.startswith('http'):
                            url = next_line
                        elif next_line.startswith('→'):
                             if i + 2 < len(lines) and lines[i+2].strip().startswith('http'):
                                 url = lines[i+2].strip()

                    res_id = hashlib.md5((title + (url or "")).encode()).hexdigest()[:8]
                    
                    # Check if we already have this resource (deduplicate)
                    if not any(n['id'] == res_id for n in nodes):
                        nodes.append({
                            "id": res_id,
                            "label": title,
                            "url": url,
                            "type": "resource",
                            "level": 3
                        })
                        edges.append({"source": current_topic, "target": res_id})
                
                i += 1
            continue

        # 5. Normal Text
        if current_node:
            buffer.append(lines[i]) # Keep original line with indentation
            
        i += 1
        
    # Flush last node
    flush_buffer()
        
    return {"nodes": nodes, "edges": edges}

def main():
    print(f"Parsing {SOURCE_FILE}...")
    os.makedirs(CURATED_DIR, exist_ok=True)
    
    graph = parse_learning_path(SOURCE_FILE)
    
    # Add Metadata
    graph['meta'] = {
        "title": "GenAI Learning Path",
        "description": "Interactive guide with self-contained recipes and resources.",
        "version": "2.0"
    }
    graph['sections'] = [] # Viewer requirement
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(graph, f, indent=2)
        
    print(f"Saved {len(graph['nodes'])} nodes to {OUTPUT_JSON}")
    print(f"Generated guides in {CURATED_DIR}")

if __name__ == "__main__":
    main()
