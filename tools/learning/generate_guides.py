#!/usr/bin/env python3
"""
Generate offline guides for content that cannot be scraped (e.g. OpenAI docs).
"""

import json
import os

# Configuration
JSON_PATH = "../../data/graphs/genai-path.json"
CURATED_DIR = "../../data/curated"

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
        .meta-header a {{ color: #58a6ff; }}
        
        /* Content Styles */
        h1, h2, h3 {{ color: #e0e0e0; margin-top: 1.5rem; }}
        p {{ line-height: 1.6; color: #c9d1d9; margin-bottom: 1rem; }}
        ul, ol {{ color: #c9d1d9; margin-bottom: 1rem; padding-left: 2rem; }}
        li {{ margin-bottom: 0.5rem; }}
        pre {{ background: #0d1117; padding: 1rem; border-radius: 6px; overflow-x: auto; margin: 1rem 0; }}
        code {{ font-family: 'Fira Code', monospace; color: #e0e0e0; }}
        .note {{ background: rgba(88, 166, 255, 0.1); border-left: 4px solid #58a6ff; padding: 1rem; margin: 1rem 0; }}
    </style>
</head>
<body>
    <div class="reader-container">
        <div class="meta-header">
            <h1>{title}</h1>
            <p>Source: <a href="{url}" target="_blank">{url}</a> (Offline Guide)</p>
        </div>
        <div class="content">
            {content}
        </div>
    </div>
</body>
</html>
"""

# Content Definitions
GUIDES = {
    "f63508c7": {
        "filename": "openai-api-docs.html",
        "content": """
        <h2>Introduction to the OpenAI API</h2>
        <p>The OpenAI API provides access to advanced AI models like GPT-4o and GPT-3.5 Turbo. It allows developers to build applications that can understand and generate natural language, code, and images.</p>
        
        <h3>Key Concepts</h3>
        <ul>
            <li><strong>Models:</strong> The AI engines. Common ones are <code>gpt-4o</code> (flagship) and <code>gpt-3.5-turbo</code> (fast/cheap).</li>
            <li><strong>Prompts:</strong> The input text you send to the model.</li>
            <li><strong>Tokens:</strong> Text is broken down into tokens (parts of words). 1000 tokens is roughly 750 words. You pay per token.</li>
            <li><strong>Context Window:</strong> The maximum number of tokens (input + output) the model can handle at once.</li>
        </ul>

        <h3>Chat Completions API</h3>
        <p>This is the main endpoint used for chat applications. It takes a list of messages as input.</p>
        <pre><code>POST https://api.openai.com/v1/chat/completions

{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
}</code></pre>

        <h3>Roles</h3>
        <ul>
            <li><strong>System:</strong> Sets the behavior/personality of the AI.</li>
            <li><strong>User:</strong> The human's input.</li>
            <li><strong>Assistant:</strong> The AI's response (used for history).</li>
        </ul>
        """
    },
    "87ac36e5": {
        "filename": "openai-structured-outputs.html",
        "content": """
        <h2>Structured Outputs (JSON Mode)</h2>
        <p>By default, LLMs return unstructured text. For building reliable applications, you often need structured data (JSON).</p>

        <h3>Why use it?</h3>
        <p>If you ask an LLM to "extract the date and time", it might say "Sure, the date is..." or just "2023-10-01". Structured outputs force it to return exactly what you need.</p>

        <h3>Using <code>response_format</code></h3>
        <p>You can enforce JSON output by setting the <code>response_format</code> parameter.</p>
        
        <pre><code>const completion = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [
    { role: "system", content: "Extract data as JSON." },
    { role: "user", content: "Meeting at 5pm on Friday." }
  ],
  response_format: { type: "json_object" }
});</code></pre>

        <div class="note">
            <strong>Important:</strong> When using JSON mode, you MUST also instruct the model in the system prompt to "output JSON".
        </div>

        <h3>Strict Schemas (Zod)</h3>
        <p>Newer models support strict schema validation, ensuring the output matches a specific structure exactly.</p>
        """
    },
    "3da90478": {
        "filename": "openai-function-calling.html",
        "content": """
        <h2>Function Calling (Tool Use)</h2>
        <p>Function calling allows the LLM to "connect" to external tools and APIs. It doesn't execute the code itself; instead, it generates the <em>arguments</em> to call a function you define.</p>

        <h3>How it works</h3>
        <ol>
            <li><strong>Define Tools:</strong> You describe your functions (e.g., <code>get_weather(location)</code>) to the model.</li>
            <li><strong>Model Decides:</strong> If the user asks "What's the weather in London?", the model sees your tool and returns a JSON object: <code>{ "name": "get_weather", "arguments": "{ 'location': 'London' }" }</code>.</li>
            <li><strong>You Execute:</strong> Your code runs the actual function/API call.</li>
            <li><strong>Return Result:</strong> You send the function's output back to the model.</li>
            <li><strong>Final Response:</strong> The model uses that data to answer the user naturally.</li>
        </ol>

        <h3>Example Definition</h3>
        <pre><code>tools = [{
  "type": "function",
  "function": {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": "string" },
        "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
      },
      "required": ["location"]
    }
  }
}]</code></pre>
        """
    },
    "86385ce2": {
        "filename": "openai-embeddings-guide.html",
        "content": """
        <h2>Embeddings Guide</h2>
        <p>Embeddings are vector representations of text. They convert words and sentences into long lists of numbers (arrays of floating-point numbers).</p>

        <h3>The Core Idea</h3>
        <p>Text with similar meanings will have similar embedding vectors. This allows us to measure "semantic similarity" mathematically.</p>
        <ul>
            <li>"The cat sits" and "The feline rests" are semantically close → High similarity score.</li>
            <li>"The cat sits" and "The stock market crashed" are far apart → Low similarity score.</li>
        </ul>

        <h3>Use Cases</h3>
        <ul>
            <li><strong>Search (RAG):</strong> Find documents relevant to a query.</li>
            <li><strong>Clustering:</strong> Group similar documents.</li>
            <li><strong>Recommendations:</strong> Find items similar to what a user likes.</li>
        </ul>

        <h3>Models</h3>
        <p>The current standard model is <code>text-embedding-3-small</code> (or <code>large</code>). It is cheaper and better than the older <code>ada-002</code>.</p>

        <h3>Cosine Similarity</h3>
        <p>The distance between two vectors is usually measured using Cosine Similarity. A value of 1.0 means identical meaning, 0 means no relation.</p>
        """
    }
}

def main():
    # Load JSON
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    nodes_map = {n['id']: n for n in data['nodes']}
    updated_count = 0

    print("Generating offline guides...")

    for node_id, guide_info in GUIDES.items():
        if node_id in nodes_map:
            node = nodes_map[node_id]
            
            # Generate HTML
            html_content = HTML_TEMPLATE.format(
                title=node['label'],
                url=node.get('url', '#'),
                content=guide_info['content']
            )
            
            # Save file
            file_path = os.path.join(CURATED_DIR, guide_info['filename'])
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Update JSON
            node['local_path'] = f"data/curated/{guide_info['filename']}"
            updated_count += 1
            print(f"  Generated: {guide_info['filename']}")
        else:
            print(f"  Warning: Node ID {node_id} not found in graph.")

    # Save updated JSON
    with open(JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nDone! Generated {updated_count} guides.")

if __name__ == "__main__":
    main()
