#!/usr/bin/env python3
"""
Fetch, clean, and curate content from the learning path resources.
Converts web pages into a clean "Reader Mode" format.
"""

import json
import os
import re
import urllib.request
import urllib.parse
from pathlib import Path
from bs4 import BeautifulSoup

# Configuration
JSON_PATH = "../../data/graphs/genai-path.json"
CURATED_DIR = "../../data/curated"
CSS_PATH = "../../assets/css/main.css"

# HTML Template for the "Reader Mode"
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
            background-color: #0f0f1a; /* Match main theme */
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
        .meta-header a {{ color: #58a6ff; }}
        
        /* Content Styles */
        img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 1rem 0; }}
        pre {{ background: #0d1117; padding: 1rem; border-radius: 6px; overflow-x: auto; }}
        code {{ font-family: 'Fira Code', monospace; color: #e0e0e0; }}
        h1, h2, h3 {{ color: #e0e0e0; margin-top: 1.5rem; }}
        p {{ line-height: 1.6; color: #c9d1d9; margin-bottom: 1rem; }}
        a {{ color: #58a6ff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        blockquote {{ border-left: 4px solid #58a6ff; padding-left: 1rem; color: #8b949e; }}
    </style>
</head>
<body>
    <div class="reader-container">
        <div class="meta-header">
            <h1>{title}</h1>
            <p>Source: <a href="{url}" target="_blank">{url}</a></p>
        </div>
        <div class="content">
            {content}
        </div>
    </div>
</body>
</html>
"""

def slugify(text):
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    return re.sub(r'[-\s]+', '-', text).strip('-')

def fetch_html(url):
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return None

def clean_content(html, base_url):
    if not html:
        return "<p>Could not fetch content.</p>"
        
    soup = BeautifulSoup(html, 'html.parser')
    
    # 1. Remove unwanted elements
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript', 'form']):
        tag.decompose()
        
    # 2. Try to find the main content area
    content = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile('content|post|article|body'))
    
    if not content:
        content = soup.body
        
    if not content:
        return "<p>Could not parse content.</p>"

    # 3. Fix relative URLs (images and links)
    for tag in content.find_all(['img', 'a']):
        if tag.name == 'img' and tag.get('src'):
            tag['src'] = urllib.parse.urljoin(base_url, tag['src'])
        if tag.name == 'a' and tag.get('href'):
            tag['href'] = urllib.parse.urljoin(base_url, tag['href'])
            tag['target'] = '_blank' # Open external links in new tab

    # 4. Remove classes and IDs to prevent style conflicts (optional, but good for consistency)
    # for tag in content.find_all(True):
    #     del tag['class']
    #     del tag['id']

    return str(content)

def main():
    # Ensure directories exist
    os.makedirs(CURATED_DIR, exist_ok=True)
    
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        
    print(f"Processing {len(data['nodes'])} nodes...")
    
    updated_count = 0
    
    for node in data['nodes']:
        # Only process 'resource' nodes that have a URL
        if node.get('type') == 'resource' and node.get('url'):
            safe_title = slugify(node['label'])
            file_name = f"{safe_title}.html"
            file_path = os.path.join(CURATED_DIR, file_name)
            
            # Skip if already exists (remove this check to force update)
            if os.path.exists(file_path):
                print(f"Skipping (exists): {node['label']}")
                node['local_path'] = f"data/curated/{file_name}"
                continue
                
            print(f"Fetching: {node['label']}...")
            raw_html = fetch_html(node['url'])
            
            if raw_html:
                clean_html = clean_content(raw_html, node['url'])
                
                final_html = HTML_TEMPLATE.format(
                    title=node['label'],
                    url=node['url'],
                    content=clean_html
                )
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(final_html)
                
                node['local_path'] = f"data/curated/{file_name}"
                updated_count += 1
                print(f"  Saved to {file_name}")
            else:
                print(f"  Failed to fetch.")

    # Save updated JSON with local paths
    with open(JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"\nDone! Updated {updated_count} resources.")

if __name__ == "__main__":
    main()
