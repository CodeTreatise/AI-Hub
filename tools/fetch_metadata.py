#!/usr/bin/env python3
"""
Fetch URL metadata (Open Graph, title, description, favicon) for all nodes.
Run this script to generate url-metadata.json for the graph viewer.

Usage: python fetch_metadata.py
"""

import json
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import sys
from pathlib import Path

# Configuration
INPUT_FILE = "../data/landscape.json"
OUTPUT_FILE = "../data/metadata.json"
TIMEOUT = 10  # seconds per request
MAX_CONCURRENT = 20  # parallel requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def extract_metadata(html: str, url: str) -> dict:
    """Extract Open Graph and meta tags from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    metadata = {
        "url": url,
        "domain": parsed_url.netloc.replace("www.", ""),
        "title": None,
        "description": None,
        "image": None,
        "favicon": None,
        "type": None,
    }
    
    # Open Graph tags (preferred)
    og_title = soup.find("meta", property="og:title")
    og_desc = soup.find("meta", property="og:description")
    og_image = soup.find("meta", property="og:image")
    og_type = soup.find("meta", property="og:type")
    
    if og_title:
        metadata["title"] = og_title.get("content", "").strip()
    if og_desc:
        metadata["description"] = og_desc.get("content", "").strip()
    if og_image:
        img = og_image.get("content", "")
        if img and not img.startswith("http"):
            img = urljoin(base_url, img)
        metadata["image"] = img
    if og_type:
        metadata["type"] = og_type.get("content", "").strip()
    
    # Twitter cards as fallback
    if not metadata["title"]:
        tw_title = soup.find("meta", {"name": "twitter:title"})
        if tw_title:
            metadata["title"] = tw_title.get("content", "").strip()
    
    if not metadata["description"]:
        tw_desc = soup.find("meta", {"name": "twitter:description"})
        if tw_desc:
            metadata["description"] = tw_desc.get("content", "").strip()
    
    if not metadata["image"]:
        tw_img = soup.find("meta", {"name": "twitter:image"})
        if tw_img:
            img = tw_img.get("content", "")
            if img and not img.startswith("http"):
                img = urljoin(base_url, img)
            metadata["image"] = img
    
    # Standard meta tags as fallback
    if not metadata["title"]:
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()
    
    if not metadata["description"]:
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc:
            metadata["description"] = meta_desc.get("content", "").strip()
    
    # Favicon
    favicon = None
    icon_link = soup.find("link", rel=lambda x: x and "icon" in x.lower() if x else False)
    if icon_link:
        favicon = icon_link.get("href", "")
        if favicon and not favicon.startswith("http"):
            favicon = urljoin(base_url, favicon)
    
    if not favicon:
        # Try default favicon location
        favicon = f"{base_url}/favicon.ico"
    
    metadata["favicon"] = favicon
    
    # Clean up description (limit length)
    if metadata["description"]:
        desc = metadata["description"]
        desc = re.sub(r'\s+', ' ', desc)  # Normalize whitespace
        if len(desc) > 200:
            desc = desc[:197] + "..."
        metadata["description"] = desc
    
    # Clean up title
    if metadata["title"]:
        title = metadata["title"]
        # Remove common suffixes
        for sep in [" | ", " - ", " – ", " — ", " :: ", " : "]:
            if sep in title:
                parts = title.split(sep)
                if len(parts[-1]) < 30:  # Likely site name
                    title = sep.join(parts[:-1])
        metadata["title"] = title.strip()
    
    return metadata


async def fetch_url(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> dict:
    """Fetch a single URL and extract metadata."""
    async with semaphore:
        try:
            headers = {"User-Agent": USER_AGENT}
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT), 
                                   headers=headers, ssl=False) as response:
                if response.status == 200:
                    html = await response.text()
                    return extract_metadata(html, url)
                else:
                    return {
                        "url": url,
                        "domain": urlparse(url).netloc.replace("www.", ""),
                        "error": f"HTTP {response.status}",
                        "title": None,
                        "description": None,
                        "image": None,
                        "favicon": f"https://{urlparse(url).netloc}/favicon.ico"
                    }
        except asyncio.TimeoutError:
            return {
                "url": url,
                "domain": urlparse(url).netloc.replace("www.", ""),
                "error": "Timeout",
                "title": None,
                "description": None,
                "image": None,
                "favicon": f"https://{urlparse(url).netloc}/favicon.ico"
            }
        except Exception as e:
            return {
                "url": url,
                "domain": urlparse(url).netloc.replace("www.", "") if url else "",
                "error": str(e)[:50],
                "title": None,
                "description": None,
                "image": None,
                "favicon": None
            }


async def main():
    """Main function to fetch all URLs."""
    # Load graph data
    input_path = Path(__file__).parent / INPUT_FILE
    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found!")
        sys.exit(1)
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Extract unique URLs
    urls = set()
    for node in data.get("nodes", []):
        url = node.get("url")
        if url and url.startswith("http"):
            urls.add(url)
    
    print(f"Found {len(urls)} unique URLs to fetch")
    
    # Load existing metadata (for incremental updates)
    output_path = Path(__file__).parent / OUTPUT_FILE
    existing_metadata = {}
    if output_path.exists():
        with open(output_path, 'r') as f:
            existing = json.load(f)
            existing_metadata = {m["url"]: m for m in existing.get("urls", [])}
        print(f"Loaded {len(existing_metadata)} existing entries")
    
    # Filter URLs that need fetching (skip already fetched without error)
    urls_to_fetch = []
    for url in urls:
        if url not in existing_metadata or existing_metadata[url].get("error"):
            urls_to_fetch.append(url)
    
    print(f"Fetching {len(urls_to_fetch)} new/failed URLs...")
    
    # Fetch URLs concurrently
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, ssl=False)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_url(session, url, semaphore) for url in urls_to_fetch]
        
        results = []
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            
            # Progress indicator
            if completed % 10 == 0 or completed == len(tasks):
                success = sum(1 for r in results if not r.get("error"))
                print(f"  Progress: {completed}/{len(tasks)} ({success} successful)")
    
    # Merge with existing
    for result in results:
        existing_metadata[result["url"]] = result
    
    # Prepare output
    output = {
        "generated": str(Path(__file__).name),
        "total_urls": len(existing_metadata),
        "successful": sum(1 for m in existing_metadata.values() if not m.get("error")),
        "urls": list(existing_metadata.values())
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Done! Saved {output['total_urls']} URLs to {OUTPUT_FILE}")
    print(f"   Successful: {output['successful']}")
    print(f"   Failed: {output['total_urls'] - output['successful']}")


if __name__ == "__main__":
    asyncio.run(main())
