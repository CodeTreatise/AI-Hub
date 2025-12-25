#!/usr/bin/env python3
"""
AI Landscape MD → JSON Parser
Extracts structured data from AI-Landscape.md Mermaid diagrams
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# === PATTERNS SPECIFIC TO AI-LANDSCAPE.MD ===

# Section header: ## 1. AI History Timeline (1950-2025)
SECTION_PATTERN = re.compile(r'^##\s+(\d+)\.\s+(.+)$')

# TOC row: | 1 | [AI History Timeline](#1-ai-history-timeline) | Description |
TOC_PATTERN = re.compile(r'^\|\s*(\d+)\s*\|\s*\[([^\]]+)\]\(#([^)]+)\)\s*\|\s*(.+?)\s*\|')

# Mermaid block
MERMAID_START = '```mermaid'
MERMAID_END = '```'

# Flowchart type: flowchart TD or flowchart LR
FLOWCHART_PATTERN = re.compile(r'flowchart\s+(TD|LR|TB|BT)')

# Node definitions:
#   NODE[<b>LABEL</b>]
#   NODE[Label]
#   NODE["Label with spaces"]
NODE_PATTERNS = [
    re.compile(r'(\w+)\[<b>([^<]+)</b>\]'),           # NODE[<b>LABEL</b>]
    re.compile(r'(\w+)\["([^"]+)"\]'),                 # NODE["Label"]
    re.compile(r'(\w+)\[([^\]]+)\]'),                  # NODE[Label]
]

# Subgraph: subgraph NAME["Display Name"]
SUBGRAPH_PATTERN = re.compile(r'subgraph\s+(\w+)\["([^"]+)"\]')

# Edges:
#   A ==> B (thick arrow)
#   A --> B (regular arrow)
#   A ==> B ==> C (chained thick)
#   A --> B[Label] (with inline node def)
EDGE_PATTERNS = [
    (re.compile(r'(\w+)\s*==>\s*(\w+)'), 'thick'),    # Thick arrow
    (re.compile(r'(\w+)\s*-->\s*(\w+)'), 'regular'),  # Regular arrow
]

# Click directive: click NODE "url" "tooltip" _blank
CLICK_PATTERN = re.compile(r'click\s+(\w+)\s+"([^"]+)"\s+"([^"]+)"\s+_blank')

# Style: style NODE fill:#xxx,stroke:#xxx,color:#xxx
STYLE_PATTERN = re.compile(r'style\s+(\w+)\s+(.+)')

# Glossary row: | **Term** | Definition |
GLOSSARY_PATTERN = re.compile(r'^\|\s*\*\*([^*]+)\*\*\s*\|\s*(.+?)\s*\|')

# Quick Reference row: | **Area** | Technologies | Roles |
QUICKREF_PATTERN = re.compile(r'^\|\s*\*\*([^*]+)\*\*\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|')


@dataclass
class Node:
    id: str
    label: str
    type: str = "node"  # node, subgraph, milestone
    section_id: Optional[int] = None
    block_id: Optional[int] = None  # Track which mermaid block in section
    url: Optional[str] = None
    tooltip: Optional[str] = None
    style: Optional[str] = None
    parent: Optional[str] = None


@dataclass
class Edge:
    source: str
    target: str
    type: str  # thick, regular
    section_id: Optional[int] = None
    block_id: Optional[int] = None  # Track which mermaid block in section


@dataclass
class Section:
    id: int
    title: str
    slug: str
    description: str
    flowchart_type: Optional[str] = None


def parse_mermaid_block(lines: list[str], section_id: int) -> tuple[list[Node], list[Edge], dict]:
    """Parse a single Mermaid code block."""
    nodes = []
    edges = []
    clicks = {}
    styles = {}
    subgraphs = []
    current_subgraph = None
    flowchart_type = None
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('%%'):
            continue
        
        # Flowchart type
        match = FLOWCHART_PATTERN.search(line)
        if match:
            flowchart_type = match.group(1)
            continue
        
        # Subgraph start
        match = SUBGRAPH_PATTERN.search(line)
        if match:
            sg_id, sg_label = match.groups()
            current_subgraph = sg_id
            nodes.append(Node(
                id=sg_id,
                label=sg_label,
                type="subgraph",
                section_id=section_id
            ))
            subgraphs.append(sg_id)
            continue
        
        # Subgraph end
        if line == 'end':
            current_subgraph = None
            continue
        
        # Click directive
        match = CLICK_PATTERN.search(line)
        if match:
            node_id, url, tooltip = match.groups()
            clicks[node_id] = {"url": url, "tooltip": tooltip}
            continue
        
        # Style
        match = STYLE_PATTERN.search(line)
        if match:
            node_id, style_def = match.groups()
            styles[node_id] = style_def
            continue
        
        # Skip linkStyle
        if line.startswith('linkStyle'):
            continue
        
        # Edges - handle various formats:
        # A --> B
        # A[Label] --> B[Label]
        # A ==> B ==> C (chained)
        
        # First find all thick edges (==>)
        if '==>' in line:
            # Split by thick arrow, handling labels in brackets
            parts = re.split(r'\s*==>\s*', line)
            for i in range(len(parts) - 1):
                # Extract node ID (first word before any bracket)
                source_match = re.search(r'(\w+)', parts[i])
                target_match = re.search(r'(\w+)', parts[i + 1])
                if source_match and target_match:
                    edges.append(Edge(
                        source=source_match.group(1),
                        target=target_match.group(1),
                        type="thick",
                        section_id=section_id
                    ))
        
        # Then find regular edges (-->) - process ALL of them
        if '-->' in line:
            # Split by regular arrow, handling labels in brackets
            parts = re.split(r'\s*-->\s*', line)
            for i in range(len(parts) - 1):
                # Extract node ID (first word before any bracket)
                source_match = re.search(r'(\w+)', parts[i])
                target_match = re.search(r'(\w+)', parts[i + 1])
                if source_match and target_match:
                    edges.append(Edge(
                        source=source_match.group(1),
                        target=target_match.group(1),
                        type="regular",
                        section_id=section_id
                    ))
        
        # Nodes (extract from edge lines too)
        for pattern in NODE_PATTERNS:
            for match in pattern.finditer(line):
                node_id, label = match.groups()
                # Skip if already added
                if not any(n.id == node_id for n in nodes):
                    nodes.append(Node(
                        id=node_id,
                        label=label,
                        type="node",
                        section_id=section_id,
                        parent=current_subgraph
                    ))
    
    # Apply clicks and styles to nodes
    for node in nodes:
        if node.id in clicks:
            node.url = clicks[node.id]["url"]
            node.tooltip = clicks[node.id]["tooltip"]
        if node.id in styles:
            node.style = styles[node.id]
    
    # Create implicit edges from subgraphs to their children
    for node in nodes:
        if node.parent and node.type == "node":
            edges.append(Edge(
                source=node.parent,
                target=node.id,
                type="contains",  # New edge type for subgraph membership
                section_id=section_id
            ))
    
    meta = {
        "flowchart_type": flowchart_type,
        "subgraphs": subgraphs
    }
    
    return nodes, edges, meta


def parse_landscape(md_path: str) -> dict:
    """Parse the entire AI-Landscape.md file."""
    
    text = Path(md_path).read_text(encoding='utf-8')
    lines = text.splitlines()
    
    result = {
        "meta": {
            "title": "AI Landscape",
            "source": md_path,
            "totalLines": len(lines),
        },
        "sections": [],
        "nodes": [],
        "edges": [],
        "glossary": [],
        "quickReference": [],
    }
    
    # === PASS 1: Extract TOC ===
    toc_map = {}
    for line in lines:
        match = TOC_PATTERN.match(line)
        if match:
            num, title, slug, desc = match.groups()
            toc_map[int(num)] = {
                "title": title,
                "slug": slug,
                "description": desc.strip()
            }
    
    # === PASS 2: Extract Sections and Mermaid blocks ===
    current_section_id = None
    mermaid_block_index = 0  # Track multiple mermaid blocks per section
    in_mermaid = False
    mermaid_lines = []
    in_glossary = False
    in_quickref = False
    
    for i, line in enumerate(lines):
        # Section header
        match = SECTION_PATTERN.match(line)
        if match:
            section_id = int(match.group(1))
            section_title = match.group(2)
            current_section_id = section_id
            
            toc_info = toc_map.get(section_id, {})
            result["sections"].append({
                "id": section_id,
                "title": section_title,
                "slug": toc_info.get("slug", ""),
                "description": toc_info.get("description", ""),
                "line": i + 1
            })
            continue
        
        # Mermaid block start
        if line.strip() == MERMAID_START:
            in_mermaid = True
            mermaid_lines = []
            mermaid_block_index += 1
            continue
        
        # Mermaid block end
        if in_mermaid and line.strip() == MERMAID_END:
            in_mermaid = False
            if current_section_id and mermaid_lines:
                # Use section_id.block_index for unique identification
                block_id = f"{current_section_id}_{mermaid_block_index}"
                nodes, edges, meta = parse_mermaid_block(mermaid_lines, current_section_id)
                # Tag nodes/edges with block_id for uniqueness
                for n in nodes:
                    n.block_id = mermaid_block_index
                for e in edges:
                    e.block_id = mermaid_block_index
                result["nodes"].extend([asdict(n) for n in nodes])
                result["edges"].extend([asdict(e) for e in edges])
            mermaid_lines = []
            continue
        
        # Inside mermaid block
        if in_mermaid:
            mermaid_lines.append(line)
            continue
        
        # Glossary section
        if "## Key Terms Glossary" in line or "| Term | Definition |" in line:
            in_glossary = True
            in_quickref = False
            continue
        
        # Quick Reference section
        if "## Quick Reference" in line:
            in_quickref = True
            in_glossary = False
            continue
        
        # Parse glossary rows
        if in_glossary:
            match = GLOSSARY_PATTERN.match(line)
            if match:
                term, definition = match.groups()
                result["glossary"].append({
                    "term": term,
                    "definition": definition.strip()
                })
        
        # Parse quick reference rows
        if in_quickref:
            match = QUICKREF_PATTERN.match(line)
            if match:
                area, tech, roles = match.groups()
                result["quickReference"].append({
                    "area": area,
                    "technologies": tech.strip(),
                    "roles": roles.strip()
                })
    
    # === Compute stats ===
    result["meta"]["totalSections"] = len(result["sections"])
    result["meta"]["totalNodes"] = len(result["nodes"])
    result["meta"]["totalEdges"] = len(result["edges"])
    result["meta"]["totalGlossaryTerms"] = len(result["glossary"])
    result["meta"]["nodesWithLinks"] = sum(1 for n in result["nodes"] if n.get("url"))
    
    return result


def build_hierarchy(nodes: list[dict], edges: list[dict]) -> dict:
    """Build parent-child hierarchy from edges."""
    hierarchy = {}
    
    # Group edges by source
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        if source not in hierarchy:
            hierarchy[source] = []
        if target not in hierarchy[source]:
            hierarchy[source].append(target)
    
    return hierarchy


def main():
    md_path = "../data/source/AI-Landscape.md"
    
    print("🔍 Parsing AI-Landscape.md...")
    result = parse_landscape(md_path)
    
    # === POST-PROCESS: Make node IDs globally unique ===
    # Prefix each node ID with section_id and block_id to avoid collisions
    id_mapping = {}  # (section_id, block_id, old_id) -> new_global_id
    
    for node in result["nodes"]:
        section_id = node.get("section_id", 0)
        block_id = node.get("block_id", 0)
        old_id = node["id"]
        new_id = f"s{section_id}_b{block_id}_{old_id}"
        
        # Store mapping for this section+block combo
        key = (section_id, block_id)
        if key not in id_mapping:
            id_mapping[key] = {}
        id_mapping[key][old_id] = new_id
        
        # Keep original ID for reference
        node["original_id"] = old_id
        node["id"] = new_id
        
        # Update parent reference
        if node.get("parent"):
            node["parent"] = f"s{section_id}_b{block_id}_{node['parent']}"
    
    # Update edge references
    for edge in result["edges"]:
        section_id = edge.get("section_id", 0)
        block_id = edge.get("block_id", 0)
        key = (section_id, block_id)
        mapping = id_mapping.get(key, {})
        if edge["source"] in mapping:
            edge["source"] = mapping[edge["source"]]
        if edge["target"] in mapping:
            edge["target"] = mapping[edge["target"]]
    
    # Add hierarchy
    result["hierarchy"] = build_hierarchy(result["nodes"], result["edges"])
    
    # Save JSON
    output_path = "../data/landscape.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved to {output_path}")
    print(f"\n📊 Stats:")
    print(f"   Sections: {result['meta']['totalSections']}")
    print(f"   Nodes: {result['meta']['totalNodes']}")
    print(f"   Edges: {result['meta']['totalEdges']}")
    print(f"   Nodes with links: {result['meta']['nodesWithLinks']}")
    print(f"   Glossary terms: {result['meta']['totalGlossaryTerms']}")
    print(f"   Quick Reference: {len(result['quickReference'])}")
    print(f"   Hierarchy entries: {len(result['hierarchy'])}")


if __name__ == "__main__":
    main()
