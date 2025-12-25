#!/usr/bin/env python3
"""Check all URLs in the AI Landscape JSON for validity."""

import json
import urllib.request
import urllib.error
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed

def main():
    # Load JSON
    with open('../data/landscape.json', 'r') as f:
        data = json.load(f)

    # Extract all unique URLs
    urls = set()
    url_to_node = {}
    for node in data['nodes']:
        if node.get('url'):
            urls.add(node['url'])
            url_to_node[node['url']] = node.get('label', node['id'])[:40]

    print(f'Total unique URLs to check: {len(urls)}')
    print('Checking URLs (this may take a minute)...')
    print()

    # Create SSL context that doesn't verify (for speed)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    def check_url(url):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response = urllib.request.urlopen(req, timeout=15, context=ctx)
            return (url, response.status, 'OK')
        except urllib.error.HTTPError as e:
            return (url, e.code, f'HTTP {e.code}')
        except urllib.error.URLError as e:
            reason = str(e.reason)[:40]
            return (url, 0, reason)
        except Exception as e:
            return (url, 0, str(e)[:40])

    # Check URLs in parallel
    results = {'ok': [], 'client_error': [], 'server_error': [], 'network_error': []}

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(check_url, url): url for url in urls}
        checked = 0
        for future in as_completed(futures):
            url, status, msg = future.result()
            checked += 1
            
            if 200 <= status < 300:
                results['ok'].append(url)
            elif 400 <= status < 500:
                results['client_error'].append((url, status, msg))
            elif status >= 500:
                results['server_error'].append((url, status, msg))
            else:
                results['network_error'].append((url, status, msg))
            
            if checked % 100 == 0:
                print(f'  Checked {checked}/{len(urls)}...')

    print()
    print('=' * 70)
    print('URL VALIDATION RESULTS')
    print('=' * 70)
    print(f'✅ OK (2xx):        {len(results["ok"])}')
    print(f'❌ Client Error (4xx): {len(results["client_error"])}')
    print(f'⚠️  Server Error (5xx): {len(results["server_error"])}')
    print(f'🔌 Network Error:     {len(results["network_error"])}')
    print()

    all_errors = results['client_error'] + results['server_error'] + results['network_error']
    
    if all_errors:
        print('FAILED URLs:')
        print('-' * 70)
        for url, status, msg in sorted(all_errors, key=lambda x: -x[1]):
            node_name = url_to_node.get(url, 'Unknown')
            print(f'[{status:3}] {msg}')
            print(f'      Node: {node_name}')
            print(f'      URL:  {url}')
            print()
        
        # Save errors to file
        with open('url_errors.txt', 'w') as f:
            f.write('URL Validation Errors\n')
            f.write('=' * 70 + '\n\n')
            for url, status, msg in sorted(all_errors, key=lambda x: -x[1]):
                node_name = url_to_node.get(url, 'Unknown')
                f.write(f'[{status}] {msg}\n')
                f.write(f'Node: {node_name}\n')
                f.write(f'URL: {url}\n\n')
        print(f'\nErrors saved to url_errors.txt')
    else:
        print('🎉 All URLs are valid!')

if __name__ == '__main__':
    main()
