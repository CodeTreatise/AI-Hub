#!/usr/bin/env python3
"""
Deep Content Fetcher for the GenAI Course.
Fetches content from multiple sources, extracts meaningful text,
follows important links, and structures it for learning.
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

# Configuration
COURSE_DIR = "../../data/course"
SOURCES_FILE = "sources.json"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

def fetch_url(url, timeout=15):
    """Fetch a URL and return the HTML content."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  ⚠️ Failed to fetch {url}: {e}")
        return None

def extract_main_content(html, base_url):
    """Extract the main readable content from HTML."""
    if not html:
        return None, []
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", 
                     "iframe", "noscript", "form", "advertisement"]):
        tag.decompose()
    
    # Try to find main content area
    content = (
        soup.find("article") or
        soup.find("main") or
        soup.find("div", class_=re.compile(r"content|post|article|body|docs|markdown", re.I)) or
        soup.find("div", id=re.compile(r"content|main|article", re.I)) or
        soup.body
    )
    
    if not content:
        return None, []
    
    # Extract internal links for crawling
    links = []
    for a in content.find_all("a", href=True):
        href = a["href"]
        # Make absolute
        full_url = urllib.parse.urljoin(base_url, href)
        # Only same-domain links
        if urllib.parse.urlparse(full_url).netloc == urllib.parse.urlparse(base_url).netloc:
            links.append({"text": a.get_text(strip=True), "url": full_url})
    
    # Fix relative URLs in content
    for tag in content.find_all(["img", "a"]):
        if tag.name == "img" and tag.get("src"):
            tag["src"] = urllib.parse.urljoin(base_url, tag["src"])
        if tag.name == "a" and tag.get("href"):
            tag["href"] = urllib.parse.urljoin(base_url, tag["href"])
            tag["target"] = "_blank"
    
    return str(content), links

def extract_text_content(html):
    """Extract plain text from HTML for analysis."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)

def extract_code_blocks(html):
    """Extract code blocks from HTML."""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    blocks = []
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        if code:
            # Try to detect language
            classes = code.get("class", [])
            lang = ""
            for cls in classes:
                if cls.startswith("language-"):
                    lang = cls.replace("language-", "")
                    break
            blocks.append({"lang": lang, "code": code.get_text()})
        else:
            blocks.append({"lang": "", "code": pre.get_text()})
    return blocks

def extract_headings(html):
    """Extract heading structure from HTML."""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    headings = []
    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        headings.append({
            "level": int(h.name[1]),
            "text": h.get_text(strip=True)
        })
    return headings

def crawl_source(source_config, max_pages=5):
    """
    Crawl a source URL and optionally follow internal links.
    Returns structured content.
    """
    url = source_config["url"]
    name = source_config.get("name", url)
    follow_links = source_config.get("follow_links", False)
    link_patterns = source_config.get("link_patterns", [])
    
    print(f"\n📥 Fetching: {name}")
    print(f"   URL: {url}")
    
    result = {
        "name": name,
        "url": url,
        "pages": []
    }
    
    # Fetch main page
    html = fetch_url(url)
    if not html:
        return result
    
    content, links = extract_main_content(html, url)
    
    result["pages"].append({
        "url": url,
        "title": name,
        "content_html": content,
        "text": extract_text_content(content),
        "code_blocks": extract_code_blocks(content),
        "headings": extract_headings(content)
    })
    
    # Follow internal links if configured
    if follow_links and links:
        visited = {url}
        to_visit = []
        
        for link in links:
            link_url = link["url"]
            # Filter by patterns if specified
            if link_patterns:
                if any(re.search(p, link_url) for p in link_patterns):
                    to_visit.append(link)
            else:
                to_visit.append(link)
        
        for link in to_visit[:max_pages - 1]:
            link_url = link["url"]
            if link_url in visited:
                continue
            visited.add(link_url)
            
            print(f"   → Following: {link['text'][:50]}...")
            time.sleep(0.5)  # Be polite
            
            sub_html = fetch_url(link_url)
            if sub_html:
                sub_content, _ = extract_main_content(sub_html, link_url)
                result["pages"].append({
                    "url": link_url,
                    "title": link["text"],
                    "content_html": sub_content,
                    "text": extract_text_content(sub_content),
                    "code_blocks": extract_code_blocks(sub_content),
                    "headings": extract_headings(sub_content)
                })
    
    print(f"   ✅ Fetched {len(result['pages'])} page(s)")
    return result

def save_source_content(module_name, source_data):
    """Save fetched content to the course directory."""
    module_dir = os.path.join(COURSE_DIR, module_name, "sources")
    os.makedirs(module_dir, exist_ok=True)
    
    # Save as JSON for processing
    safe_name = re.sub(r'[^\w\-]', '_', source_data["name"].lower())
    filepath = os.path.join(module_dir, f"{safe_name}.json")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(source_data, f, indent=2, ensure_ascii=False)
    
    return filepath

def fetch_module_sources(module_name, sources):
    """Fetch all sources for a module."""
    print(f"\n{'='*60}")
    print(f"📚 Fetching sources for: {module_name}")
    print(f"{'='*60}")
    
    results = []
    for source in sources:
        data = crawl_source(source)
        if data["pages"]:
            filepath = save_source_content(module_name, data)
            results.append({"name": source["name"], "file": filepath, "pages": len(data["pages"])})
    
    return results

# ============================================================
# Module Source Definitions
# ============================================================

MODULE_SOURCES = {
    "module-0-overview": [
        {
            "name": "What is RAG - Pinecone",
            "url": "https://www.pinecone.io/learn/retrieval-augmented-generation/",
            "follow_links": False
        },
        {
            "name": "LLM Introduction - Hugging Face",
            "url": "https://huggingface.co/docs/transformers/llm_tutorial",
            "follow_links": False
        }
    ],
    "module-1-docker": [
        {
            "name": "Docker Overview",
            "url": "https://docs.docker.com/get-started/overview/",
            "follow_links": True,
            "link_patterns": [r"/get-started/"]
        },
        {
            "name": "Docker Compose Getting Started",
            "url": "https://docs.docker.com/compose/gettingstarted/",
            "follow_links": False
        }
    ],
    "module-2-postgres": [
        {
            "name": "PostgreSQL Tutorial",
            "url": "https://www.postgresqltutorial.com/postgresql-getting-started/",
            "follow_links": True,
            "link_patterns": [r"postgresql-tutorial", r"getting-started"]
        },
        {
            "name": "pgvector GitHub",
            "url": "https://github.com/pgvector/pgvector",
            "follow_links": False
        },
        {
            "name": "Supabase pgvector Guide",
            "url": "https://supabase.com/docs/guides/ai/vector-columns",
            "follow_links": False
        }
    ],
    "module-3-llm": [
        {
            "name": "OpenAI Cookbook",
            "url": "https://cookbook.openai.com/",
            "follow_links": False
        },
        {
            "name": "Prompt Engineering Guide",
            "url": "https://www.promptingguide.ai/",
            "follow_links": True,
            "link_patterns": [r"/introduction", r"/techniques"]
        }
    ],
    "module-4-embeddings": [
        {
            "name": "What are Embeddings - Pinecone",
            "url": "https://www.pinecone.io/learn/vector-embeddings/",
            "follow_links": False
        },
        {
            "name": "Embeddings Guide - Hugging Face",
            "url": "https://huggingface.co/blog/getting-started-with-embeddings",
            "follow_links": False
        }
    ],
    "module-5-rag": [
        {
            "name": "RAG Tutorial - LangChain",
            "url": "https://python.langchain.com/docs/tutorials/rag/",
            "follow_links": False
        },
        {
            "name": "Chunking Strategies - Pinecone",
            "url": "https://www.pinecone.io/learn/chunking-strategies/",
            "follow_links": False
        }
    ]
}

def main():
    """Fetch all module sources."""
    os.makedirs(COURSE_DIR, exist_ok=True)
    
    all_results = {}
    
    for module_name, sources in MODULE_SOURCES.items():
        results = fetch_module_sources(module_name, sources)
        all_results[module_name] = results
    
    # Save summary
    summary_path = os.path.join(COURSE_DIR, "fetch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ All sources fetched!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
