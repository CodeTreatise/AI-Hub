#!/usr/bin/env python3
"""
Validate extracted JSON against source AI-Landscape.md
"""

import re
import json
from pathlib import Path
from collections import Counter

def load_data():
    md_path = "../data/source/AI-Landscape.md"
    json_path = "../data/landscape.json"
    
    md_text = Path(md_path).read_text(encoding='utf-8')
    with open(json_path) as f:
        json_data = json.load(f)
    
    return md_text, json_data

def extract_mermaid_blocks(md_text):
    """Extract all mermaid code blocks from MD."""
    blocks = []
    in_mermaid = False
    current_block = []
    
    for line in md_text.splitlines():
        if line.strip() == '```mermaid':
            in_mermaid = True
            current_block = []
        elif in_mermaid and line.strip() == '```':
            in_mermaid = False
            blocks.append('\n'.join(current_block))
        elif in_mermaid:
            current_block.append(line)
    
    return blocks

def count_in_md(md_text):
    """Count elements directly from MD file."""
    counts = {}
    
    # Section headers
    counts['sections'] = len(re.findall(r'^##\s+\d+\.', md_text, re.MULTILINE))
    
    # Mermaid blocks
    blocks = extract_mermaid_blocks(md_text)
    counts['mermaid_blocks'] = len(blocks)
    
    # Click directives
    clicks = re.findall(r'click\s+(\w+)\s+"([^"]+)"\s+"([^"]+)"\s+_blank', md_text)
    counts['click_directives'] = len(clicks)
    counts['unique_click_nodes'] = len(set(c[0] for c in clicks))
    
    # Edge patterns in mermaid
    all_mermaid = '\n'.join(blocks)
    thick_edges = re.findall(r'(\w+)\s*==>\s*(\w+)', all_mermaid)
    regular_edges = re.findall(r'(\w+)\s*-->\s*(\w+)', all_mermaid)
    counts['thick_edges'] = len(thick_edges)
    counts['regular_edges'] = len(regular_edges)
    counts['total_edges'] = len(thick_edges) + len(regular_edges)
    
    # Node definitions
    node_patterns = [
        r'(\w+)\[<b>([^<]+)</b>\]',
        r'(\w+)\["([^"]+)"\]',
        r'subgraph\s+(\w+)\["([^"]+)"\]',
    ]
    all_nodes = set()
    for pattern in node_patterns:
        for match in re.finditer(pattern, all_mermaid):
            all_nodes.add(match.group(1))
    
    # Also get nodes from edges (left side and right side)
    for src, tgt in thick_edges + regular_edges:
        all_nodes.add(src)
        all_nodes.add(tgt)
    
    counts['unique_nodes_in_mermaid'] = len(all_nodes)
    
    # Glossary entries
    glossary_matches = re.findall(r'^\|\s*\*\*([^*]+)\*\*\s*\|', md_text, re.MULTILINE)
    counts['glossary_terms'] = len(glossary_matches)
    
    return counts, clicks, thick_edges, regular_edges, all_nodes

def validate(md_text, json_data):
    """Compare MD counts vs JSON counts."""
    
    md_counts, md_clicks, thick_edges, regular_edges, md_nodes = count_in_md(md_text)
    
    print("=" * 60)
    print("VALIDATION: MD vs JSON")
    print("=" * 60)
    
    comparisons = [
        ("Sections", md_counts['sections'], json_data['meta']['totalSections']),
        ("Mermaid Blocks", md_counts['mermaid_blocks'], 49),  # From earlier analysis
        ("Click Directives", md_counts['click_directives'], json_data['meta']['nodesWithLinks']),
        ("Total Edges", md_counts['total_edges'], json_data['meta']['totalEdges']),
        ("Glossary Terms", md_counts['glossary_terms'], json_data['meta']['totalGlossaryTerms']),
    ]
    
    all_pass = True
    for name, md_val, json_val in comparisons:
        status = "✅" if md_val == json_val else "❌"
        if md_val != json_val:
            all_pass = False
        print(f"{status} {name}: MD={md_val}, JSON={json_val}")
    
    print("\n" + "=" * 60)
    print("EDGE TYPE BREAKDOWN")
    print("=" * 60)
    print(f"Thick edges (==>): {md_counts['thick_edges']}")
    print(f"Regular edges (-->): {md_counts['regular_edges']}")
    print(f"Total: {md_counts['total_edges']}")
    
    # Verify in JSON
    json_thick = sum(1 for e in json_data['edges'] if e['type'] == 'thick')
    json_regular = sum(1 for e in json_data['edges'] if e['type'] == 'regular')
    print(f"\nJSON thick: {json_thick}, JSON regular: {json_regular}")
    
    print("\n" + "=" * 60)
    print("EDGE SYNTAX EXAMPLES FROM MD")
    print("=" * 60)
    
    blocks = extract_mermaid_blocks(md_text)
    
    print("\n### Thick Arrows (==>) - Main flow connections:")
    examples = []
    for block in blocks[:3]:
        for line in block.splitlines():
            if '==>' in line and len(examples) < 5:
                examples.append(line.strip())
    for ex in examples:
        print(f"  {ex}")
    
    print("\n### Regular Arrows (-->) - Detail connections:")
    examples = []
    for block in blocks[:5]:
        for line in block.splitlines():
            if '-->' in line and '==>' not in line and len(examples) < 5:
                examples.append(line.strip())
    for ex in examples:
        print(f"  {ex}")
    
    print("\n" + "=" * 60)
    print("CONNECTION PATTERNS IN YOUR MD")
    print("=" * 60)
    
    print("""
    PATTERN 1: Main Category Flow (thick arrows)
    ┌─────────────────────────────────────────┐
    │  MAIN[<b>MAIN TOPIC</b>]                │
    │                                         │
    │  MAIN ==> CAT1[<b>Category 1</b>]       │
    │  MAIN ==> CAT2[<b>Category 2</b>]       │
    │  MAIN ==> CAT3[<b>Category 3</b>]       │
    └─────────────────────────────────────────┘
    
    PATTERN 2: Detail Items (regular arrows)
    ┌─────────────────────────────────────────┐
    │  CAT1 --> ITEM1[Item 1]                 │
    │  CAT1 --> ITEM2[Item 2]                 │
    │  CAT1 --> ITEM3[Item 3]                 │
    └─────────────────────────────────────────┘
    
    PATTERN 3: Subgraphs for grouping
    ┌─────────────────────────────────────────┐
    │  subgraph GROUP["Group Name"]           │
    │      A[Item A]                          │
    │      B[Item B]                          │
    │  end                                    │
    └─────────────────────────────────────────┘
    
    PATTERN 4: Timeline/Sequential (thick arrows)
    ┌─────────────────────────────────────────┐
    │  Birth ==> Winter1 ==> Revival ==> DL  │
    └─────────────────────────────────────────┘
    """)
    
    # Check for missing clicks
    print("\n" + "=" * 60)
    print("CLICK COVERAGE ANALYSIS")
    print("=" * 60)
    
    clicked_nodes = set(c[0] for c in md_clicks)
    nodes_without_clicks = md_nodes - clicked_nodes
    
    print(f"Nodes defined in Mermaid: {len(md_nodes)}")
    print(f"Nodes with click directives: {len(clicked_nodes)}")
    print(f"Nodes missing clicks: {len(nodes_without_clicks)}")
    
    if nodes_without_clicks:
        print(f"\nMissing clicks (first 20): {list(nodes_without_clicks)[:20]}")
    
    # Check edges by section
    print("\n" + "=" * 60)
    print("EDGES BY SECTION (from JSON)")
    print("=" * 60)
    
    edges_by_section = Counter(e['section_id'] for e in json_data['edges'])
    for section in json_data['sections'][:10]:
        sid = section['id']
        count = edges_by_section.get(sid, 0)
        print(f"  Section {sid}: {section['title'][:40]:<40} → {count} edges")
    
    print(f"  ... and {len(json_data['sections']) - 10} more sections")
    
    return all_pass

def show_hierarchy_sample(json_data):
    """Show hierarchy structure."""
    print("\n" + "=" * 60)
    print("HIERARCHY STRUCTURE (How nodes connect)")
    print("=" * 60)
    
    # Find key hierarchies
    key_roots = ['AI', 'ML', 'DL', 'TR', 'LLM', 'GENAI', 'RAG', 'AGENT']
    
    for root in key_roots:
        if root in json_data['hierarchy']:
            children = json_data['hierarchy'][root]
            print(f"\n{root} →")
            for child in children[:8]:
                print(f"  ├── {child}")
                if child in json_data['hierarchy']:
                    grandchildren = json_data['hierarchy'][child][:3]
                    for gc in grandchildren:
                        print(f"  │   └── {gc}")
            if len(children) > 8:
                print(f"  └── ... and {len(children) - 8} more")

def main():
    md_text, json_data = load_data()
    
    validate(md_text, json_data)
    show_hierarchy_sample(json_data)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Your Mermaid diagrams use TWO edge types:

1. THICK ARROWS (==>) 
   - Connect main categories
   - Show primary flow/hierarchy
   - Example: MAIN ==> CATEGORY
   
2. REGULAR ARROWS (-->)
   - Connect categories to details
   - Show secondary relationships
   - Example: CATEGORY --> item

This creates a 2-level hierarchy in most diagrams:
   
   MAIN TOPIC (root)
      ├══► Category A (thick)
      │      ├──► Detail 1 (regular)
      │      └──► Detail 2 (regular)
      └══► Category B (thick)
             ├──► Detail 3 (regular)
             └──► Detail 4 (regular)
    """)

if __name__ == "__main__":
    main()
