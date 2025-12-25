import json
import os

path = "/workspace/Learning/Gen-Ai/graph-viewer/data/graphs/genai-path.json"

print(f"Reading {path}...")
try:
    with open(path, 'r') as f:
        data = json.load(f)
    
    print(f"Keys before: {list(data.keys())}")

    data['meta'] = {
        "title": "GenAI Learning Path",
        "description": "A structured guide to mastering Generative AI and RAG development.",
        "version": "1.0"
    }
    
    if 'sections' not in data:
        data['sections'] = []
        
    print(f"Keys after: {list(data.keys())}")

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print("Successfully saved file.")

except Exception as e:
    print(f"Error: {e}")
