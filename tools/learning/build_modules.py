#!/usr/bin/env python3
"""
Generate course modules from fetched sources.
Combines source content with our teaching structure.
"""

import os
import re
import json
from pathlib import Path
from bs4 import BeautifulSoup

COURSE_DIR = "../../data/course"

# HTML Template for module pages
MODULE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | GenAI Course</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0f0f1a;
            color: #c9d1d9;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        /* Header */
        .module-header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }}
        
        .module-header h1 {{
            font-size: 2rem;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        
        .module-meta {{
            display: flex;
            gap: 1.5rem;
            color: #8b949e;
            font-size: 0.9rem;
            margin-top: 1rem;
        }}
        
        .module-meta span {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        /* Sections */
        .section {{
            background: #1a1a2e;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            border: 1px solid #30363d;
        }}
        
        .section h2 {{
            color: #e0e0e0;
            font-size: 1.4rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .section h3 {{
            color: #c9d1d9;
            font-size: 1.1rem;
            margin: 1.5rem 0 0.75rem;
        }}
        
        .section p {{
            margin-bottom: 1rem;
        }}
        
        .section ul, .section ol {{
            padding-left: 1.5rem;
            margin-bottom: 1rem;
        }}
        
        .section li {{
            margin-bottom: 0.5rem;
        }}
        
        /* Objectives */
        .objectives ul {{
            list-style: none;
            padding: 0;
        }}
        
        .objectives li {{
            padding: 0.5rem 0;
            padding-left: 1.5rem;
            position: relative;
        }}
        
        .objectives li::before {{
            content: "✓";
            position: absolute;
            left: 0;
            color: #10b981;
        }}
        
        /* Prerequisites */
        .prerequisites {{
            background: rgba(88, 166, 255, 0.1);
            border-left: 4px solid #58a6ff;
        }}
        
        /* Code blocks */
        pre {{
            background: #0d1117;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            margin: 1rem 0;
            border: 1px solid #30363d;
        }}
        
        code {{
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.9rem;
        }}
        
        :not(pre) > code {{
            background: rgba(110, 118, 129, 0.4);
            padding: 0.2em 0.4em;
            border-radius: 6px;
        }}
        
        /* Callouts */
        .callout {{
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }}
        
        .callout-tip {{
            background: rgba(16, 185, 129, 0.1);
            border-left: 4px solid #10b981;
        }}
        
        .callout-warning {{
            background: rgba(245, 158, 11, 0.1);
            border-left: 4px solid #f59e0b;
        }}
        
        .callout-info {{
            background: rgba(59, 130, 246, 0.1);
            border-left: 4px solid #3b82f6;
        }}
        
        /* Quiz */
        .quiz-question {{
            background: #161b22;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }}
        
        .quiz-question p {{
            font-weight: 600;
            margin-bottom: 0.75rem;
        }}
        
        .quiz-options {{
            list-style: none;
            padding: 0;
        }}
        
        .quiz-options li {{
            padding: 0.5rem 1rem;
            margin: 0.25rem 0;
            background: #0d1117;
            border-radius: 6px;
            cursor: pointer;
            border: 1px solid transparent;
        }}
        
        .quiz-options li:hover {{
            border-color: #58a6ff;
        }}
        
        /* References */
        .references ul {{
            list-style: none;
            padding: 0;
        }}
        
        .references li {{
            padding: 0.75rem;
            background: #161b22;
            border-radius: 6px;
            margin: 0.5rem 0;
        }}
        
        .references a {{
            color: #58a6ff;
            text-decoration: none;
        }}
        
        .references a:hover {{
            text-decoration: underline;
        }}

        /* Top nav (unified routing) */
        .top-nav {{
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }}

        .top-nav a {{
            color: #58a6ff;
            text-decoration: none;
            padding: 0.5rem 0.9rem;
            background: #161b22;
            border-radius: 999px;
            border: 1px solid #30363d;
            font-size: 0.9rem;
        }}

        .top-nav a:hover {{
            background: #1a1a2e;
            border-color: #58a6ff;
        }}
        
        /* Navigation */
        .nav-links {{
            display: flex;
            justify-content: space-between;
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid #30363d;
        }}
        
        .nav-links a {{
            color: #58a6ff;
            text-decoration: none;
            padding: 0.75rem 1.5rem;
            background: #161b22;
            border-radius: 8px;
            border: 1px solid #30363d;
        }}
        
        .nav-links a:hover {{
            background: #1a1a2e;
            border-color: #58a6ff;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .container {{ padding: 1rem; }}
            .section {{ padding: 1.5rem; }}
            .module-meta {{ flex-wrap: wrap; gap: 0.75rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <nav class="top-nav">
            <a href="../../../index.html" target="_top">🏠 Hub</a>
            <a href="../../../pages/study.html" target="_top">📖 Study Mode</a>
            <a href="../index.html" target="_top">📚 Course Home</a>
        </nav>
        <header class="module-header">
            <h1>{title}</h1>
            <p>{description}</p>
            <div class="module-meta">
                <span>⏱️ {time_estimate}</span>
                <span>📊 {difficulty}</span>
                <span>📚 Module {module_number}</span>
            </div>
        </header>
        
        {content}
        
        <nav class="nav-links">
            {prev_link}
            {next_link}
        </nav>
    </div>
</body>
</html>
"""

def load_source_content(module_dir):
    """Load all source JSON files for a module."""
    sources_dir = os.path.join(module_dir, "sources")
    if not os.path.exists(sources_dir):
        return []
    
    sources = []
    for filename in os.listdir(sources_dir):
        if filename.endswith(".json"):
            with open(os.path.join(sources_dir, filename), "r") as f:
                sources.append(json.load(f))
    return sources

def extract_key_concepts(sources):
    """Extract key concepts from source content."""
    concepts = []
    for source in sources:
        for page in source.get("pages", []):
            for heading in page.get("headings", []):
                if heading["level"] <= 3:
                    concepts.append(heading["text"])
    return concepts

def build_section(section_type, title, content, icon="📖"):
    """Build an HTML section."""
    return f"""
        <section class="section {section_type}">
            <h2>{icon} {title}</h2>
            {content}
        </section>
    """

def build_module_0():
    """Build Module 0: Why GenAI/RAG - The Big Picture"""
    
    sources = load_source_content(os.path.join(COURSE_DIR, "module-0-overview"))
    
    content = ""
    
    # Objectives
    objectives = """
        <ul>
            <li>Understand what Generative AI is and why it matters</li>
            <li>Know the difference between traditional AI and GenAI</li>
            <li>Understand what RAG is and why we need it</li>
            <li>See the big picture of what you'll build</li>
            <li>Understand your career path as a GenAI developer</li>
        </ul>
    """
    content += build_section("objectives", "What You'll Learn", objectives, "🎯")
    
    # The Big Picture
    big_picture = """
        <h3>What is Generative AI?</h3>
        <p>Generative AI refers to artificial intelligence systems that can <strong>create new content</strong> - text, images, code, music, and more. Unlike traditional AI that classifies or predicts, GenAI <em>generates</em>.</p>
        
        <div class="callout callout-info">
            <strong>Simple analogy:</strong> Traditional AI is like a librarian who finds books for you. Generative AI is like an author who writes new books based on everything they've read.
        </div>
        
        <h3>Large Language Models (LLMs)</h3>
        <p>LLMs are the engines behind text-based GenAI. They've been trained on massive amounts of text and can:</p>
        <ul>
            <li>Answer questions</li>
            <li>Write code</li>
            <li>Summarize documents</li>
            <li>Translate languages</li>
            <li>Have conversations</li>
        </ul>
        
        <p>Examples: GPT-4, Claude, Llama, Gemini</p>
        
        <h3>The Problem with LLMs</h3>
        <p>LLMs have a critical limitation: <strong>they only know what they were trained on</strong>. They can't:</p>
        <ul>
            <li>Access your private documents</li>
            <li>Know about events after their training cutoff</li>
            <li>Answer questions about your specific business data</li>
        </ul>
        
        <div class="callout callout-warning">
            <strong>This is why we need RAG!</strong> Without RAG, LLMs are like brilliant experts who've been locked in a room since 2023 with no access to your company's information.
        </div>
    """
    content += build_section("content", "The Big Picture", big_picture, "📖")
    
    # What is RAG
    rag_section = """
        <h3>RAG = Retrieval-Augmented Generation</h3>
        <p>RAG is a technique that gives LLMs access to external knowledge. Here's how it works:</p>
        
        <ol>
            <li><strong>User asks a question</strong> - "What's our refund policy?"</li>
            <li><strong>Retrieval</strong> - System searches your documents and finds relevant info</li>
            <li><strong>Augmentation</strong> - The found information is added to the LLM's prompt</li>
            <li><strong>Generation</strong> - LLM generates an answer using that context</li>
        </ol>
        
        <div class="callout callout-tip">
            <strong>Think of it like this:</strong> Instead of asking someone a question from memory, you hand them the relevant documents first and THEN ask the question.
        </div>
        
        <h3>Why RAG is Powerful</h3>
        <ul>
            <li><strong>Up-to-date:</strong> Your documents can change anytime</li>
            <li><strong>Private:</strong> Your data never leaves your control</li>
            <li><strong>Accurate:</strong> Answers are grounded in real documents</li>
            <li><strong>Verifiable:</strong> You can show which documents were used</li>
            <li><strong>Cost-effective:</strong> No need to retrain expensive models</li>
        </ul>
    """
    content += build_section("content", "What is RAG?", rag_section, "🔍")
    
    # What You'll Build
    build_section_content = """
        <p>In this course, you'll build a complete RAG application:</p>
        
        <pre><code>┌─────────────────────────────────────────────────────────┐
│                    YOUR RAG APP                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   📄 Documents ──▶ 🔪 Chunker ──▶ 🔢 Embeddings         │
│                                          │               │
│                                          ▼               │
│                                    ┌──────────┐          │
│                                    │ pgvector │          │
│                                    │ Database │          │
│                                    └──────────┘          │
│                                          │               │
│   💬 User Query ──▶ 🔢 Embed ──▶ 🔍 Search │            │
│                                          │               │
│                                          ▼               │
│                                    ┌──────────┐          │
│                     Context ◀───── │ Results  │          │
│                        │           └──────────┘          │
│                        ▼                                 │
│                   ┌─────────┐                            │
│                   │  LLM    │ ──▶ 💬 Answer             │
│                   │ (GPT-4) │                            │
│                   └─────────┘                            │
│                                                          │
└─────────────────────────────────────────────────────────┘</code></pre>
        
        <h3>Your Tech Stack</h3>
        <ul>
            <li><strong>Backend:</strong> Node.js + Express + TypeScript</li>
            <li><strong>Database:</strong> PostgreSQL + pgvector</li>
            <li><strong>LLM:</strong> OpenAI API (GPT-4)</li>
            <li><strong>Frontend:</strong> Angular (your existing skills!)</li>
            <li><strong>DevOps:</strong> Docker for local development</li>
        </ul>
    """
    content += build_section("content", "What You'll Build", build_section_content, "🛠️")
    
    # Why This Matters - Career
    career_section = """
        <h3>The Demand for GenAI Developers</h3>
        <p>As of 2024-2025, GenAI skills are among the most in-demand in tech. Companies are racing to integrate AI into their products, and they need developers who understand:</p>
        <ul>
            <li>How to build practical AI applications (not just use ChatGPT)</li>
            <li>How to make AI work with private/company data (RAG)</li>
            <li>How to deploy and scale these systems</li>
        </ul>
        
        <h3>Your Advantage</h3>
        <p>You already know Node.js and Angular. That's a huge head start:</p>
        <ul>
            <li>Most RAG tutorials are in Python - you'll stand out with TypeScript expertise</li>
            <li>You can build full-stack AI apps, not just backend</li>
            <li>Your Angular skills apply directly to building chat UIs</li>
        </ul>
        
        <div class="callout callout-tip">
            <strong>Tip:</strong> The goal isn't to become an ML researcher. It's to become a developer who can <em>apply</em> AI to solve real problems.
        </div>
    """
    content += build_section("content", "Why This Matters For Your Career", career_section, "💼")
    
    # Quiz
    quiz = """
        <div class="quiz-question">
            <p>1. What does RAG stand for?</p>
            <ul class="quiz-options">
                <li>A) Random Access Generation</li>
                <li>B) Retrieval-Augmented Generation</li>
                <li>C) Rapid AI Growth</li>
                <li>D) Real-time Answer Generator</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>2. What problem does RAG solve?</p>
            <ul class="quiz-options">
                <li>A) LLMs are too slow</li>
                <li>B) LLMs don't know about your private data</li>
                <li>C) LLMs are too expensive</li>
                <li>D) LLMs can't generate text</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>3. In a RAG system, what happens BEFORE the LLM generates an answer?</p>
            <ul class="quiz-options">
                <li>A) The LLM is retrained on new data</li>
                <li>B) Relevant documents are retrieved and added to the prompt</li>
                <li>C) The user's question is translated to another language</li>
                <li>D) The database is backed up</li>
            </ul>
        </div>
        
        <p style="margin-top: 1rem; color: #8b949e;"><em>Answers: 1-B, 2-B, 3-B</em></p>
    """
    content += build_section("quiz", "Check Your Understanding", quiz, "✅")
    
    # References
    refs = """
        <ul>
            <li>
                <a href="https://www.pinecone.io/learn/retrieval-augmented-generation/" target="_blank">
                    What is RAG? - Pinecone
                </a>
                <span style="color: #8b949e;"> - Comprehensive overview of RAG</span>
            </li>
            <li>
                <a href="https://huggingface.co/docs/transformers/llm_tutorial" target="_blank">
                    LLM Introduction - Hugging Face
                </a>
                <span style="color: #8b949e;"> - Understanding large language models</span>
            </li>
            <li>
                <a href="https://platform.openai.com/docs" target="_blank">
                    OpenAI Documentation
                </a>
                <span style="color: #8b949e;"> - Official API docs</span>
            </li>
        </ul>
    """
    content += build_section("references", "Sources & Further Reading", refs, "📚")
    
    # Generate HTML
    html = MODULE_TEMPLATE.format(
        title="Module 0: The Big Picture",
        description="Understanding GenAI, LLMs, and RAG - and why this matters for your career",
        time_estimate="~30 minutes",
        difficulty="Beginner",
        module_number="0",
        content=content,
        prev_link='<a href="../index.html">← Course Home</a>',
        next_link='<a href="../module-1-docker/index.html">Next: Dev Environment →</a>'
    )
    
    # Save
    output_dir = os.path.join(COURSE_DIR, "module-0-overview")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
    
    print("✅ Generated Module 0: The Big Picture")

def build_module_1():
    """Build Module 1: Docker & Development Environment"""
    
    content = ""
    
    # Objectives
    objectives = """
        <ul>
            <li>Understand what Docker is and why we use it</li>
            <li>Know the difference between containers and virtual machines</li>
            <li>Understand Docker images, containers, and Dockerfiles</li>
            <li>Master Docker Compose for multi-container applications</li>
            <li>Set up your complete development environment for this course</li>
        </ul>
    """
    content += build_section("objectives", "What You'll Learn", objectives, "🎯")
    
    # Prerequisites
    prereqs = """
        <p>Before starting this module, make sure you have:</p>
        <ul>
            <li>A computer running Linux, macOS, or Windows</li>
            <li>Admin/sudo access to install software</li>
            <li>Basic command line knowledge (cd, ls, mkdir)</li>
            <li>A code editor (VS Code recommended)</li>
        </ul>
        <div class="callout callout-info">
            <strong>Don't worry!</strong> We'll explain everything step by step. Even if you've never used Docker, you'll be comfortable with it by the end of this module.
        </div>
    """
    content += build_section("prerequisites", "Prerequisites", prereqs, "📋")
    
    # What is Docker
    docker_intro = """
        <h3>The Problem Docker Solves</h3>
        <p>Have you ever heard (or said) "It works on my machine!"? This happens because:</p>
        <ul>
            <li>Different operating systems behave differently</li>
            <li>Different versions of software (Node.js, Python, PostgreSQL) may be installed</li>
            <li>Environment variables and configurations differ</li>
            <li>Some dependencies are missing on other machines</li>
        </ul>
        
        <div class="callout callout-warning">
            <strong>Real-world problem:</strong> You build an app with PostgreSQL 16 on your Mac. Your teammate has PostgreSQL 14 on Ubuntu. The production server runs PostgreSQL 15 on Amazon Linux. Things break in mysterious ways.
        </div>
        
        <h3>Docker's Solution: Containers</h3>
        <p>Docker packages your application AND its entire environment (OS, dependencies, configurations) into a <strong>container</strong>. This container runs exactly the same everywhere.</p>
        
        <blockquote style="border-left: 4px solid #818cf8; padding-left: 1rem; color: #c9d1d9; margin: 1rem 0;">
            "Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly."
            <footer style="color: #8b949e; margin-top: 0.5rem;">— Docker Documentation</footer>
        </blockquote>
        
        <h3>Container vs Virtual Machine</h3>
        <p>You might be wondering: "Isn't this like a virtual machine?" Not quite:</p>
        
        <pre><code>┌─────────────────────────────────────────────────────────────┐
│                    VIRTUAL MACHINES                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │
│  │  App 1  │  │  App 2  │  │  App 3  │  ← Your apps         │
│  ├─────────┤  ├─────────┤  ├─────────┤                      │
│  │Guest OS │  │Guest OS │  │Guest OS │  ← FULL OS each!     │
│  │(Ubuntu) │  │(CentOS) │  │(Debian) │     (heavy: ~GB)     │
│  └─────────┘  └─────────┘  └─────────┘                      │
│  ┌───────────────────────────────────────┐                  │
│  │           Hypervisor (VMware)          │                  │
│  └───────────────────────────────────────┘                  │
│  ┌───────────────────────────────────────┐                  │
│  │           Host OS (Your Machine)       │                  │
│  └───────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      CONTAINERS                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │
│  │  App 1  │  │  App 2  │  │  App 3  │  ← Your apps         │
│  ├─────────┤  ├─────────┤  ├─────────┤                      │
│  │  Libs   │  │  Libs   │  │  Libs   │  ← Just libraries    │
│  │  Only   │  │  Only   │  │  Only   │     (light: ~MB)     │
│  └─────────┘  └─────────┘  └─────────┘                      │
│  ┌───────────────────────────────────────┐                  │
│  │           Docker Engine                │                  │
│  └───────────────────────────────────────┘                  │
│  ┌───────────────────────────────────────┐                  │
│  │     Host OS (Shared Linux Kernel)      │                  │
│  └───────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘</code></pre>
        
        <p><strong>Key differences:</strong></p>
        <ul>
            <li><strong>Size:</strong> VMs are gigabytes, containers are megabytes</li>
            <li><strong>Startup:</strong> VMs take minutes, containers take seconds</li>
            <li><strong>Resources:</strong> VMs need dedicated RAM/CPU, containers share host resources</li>
            <li><strong>Isolation:</strong> VMs are fully isolated, containers share the host kernel</li>
        </ul>
        
        <div class="callout callout-tip">
            <strong>Simple analogy:</strong> VMs are like houses (each with its own plumbing, electricity, foundation). Containers are like apartments (shared building infrastructure, but isolated living spaces).
        </div>
    """
    content += build_section("content", "What is Docker?", docker_intro, "🐳")
    
    # Docker Architecture
    architecture = """
        <h3>Core Concepts</h3>
        <p>Docker has three main components you need to understand:</p>
        
        <h4>1. Docker Image</h4>
        <p>An <strong>image</strong> is a read-only template with instructions for creating a container. Think of it like a recipe or a blueprint.</p>
        <ul>
            <li>Images are built from a <code>Dockerfile</code></li>
            <li>Images are stored in registries (like Docker Hub)</li>
            <li>Images are layered (each instruction adds a layer)</li>
            <li>Images can be based on other images</li>
        </ul>
        
        <pre><code># Example: This image is based on node:18, then adds your app
FROM node:18-alpine
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
CMD ["npm", "start"]</code></pre>
        
        <h4>2. Docker Container</h4>
        <p>A <strong>container</strong> is a runnable instance of an image. It's like a running instance of your recipe.</p>
        <ul>
            <li>You can create, start, stop, move, or delete containers</li>
            <li>Containers are isolated from each other</li>
            <li>You can run multiple containers from the same image</li>
            <li>Changes to a container don't affect the image</li>
        </ul>
        
        <h4>3. Docker Daemon & Client</h4>
        <p>The <strong>Docker daemon</strong> (<code>dockerd</code>) does the heavy lifting - building, running, and managing containers. The <strong>Docker client</strong> (<code>docker</code>) is the command-line tool you use.</p>
        
        <pre><code># The docker command (client) talks to dockerd (daemon)
$ docker run nginx     # Client sends "run nginx" to daemon
$ docker ps            # Client asks daemon "what's running?"
$ docker stop abc123   # Client tells daemon to stop container</code></pre>
        
        <div class="callout callout-info">
            <strong>Docker Desktop</strong> is an application that installs both the daemon and client, plus a nice GUI. It's the easiest way to get started on Mac/Windows.
        </div>
    """
    content += build_section("content", "Docker Architecture", architecture, "🏗️")
    
    # Installing Docker
    install = """
        <h3>Step 1: Install Docker Desktop</h3>
        <p>The easiest way to get Docker is via Docker Desktop:</p>
        
        <ol>
            <li>Go to <a href="https://www.docker.com/products/docker-desktop/" target="_blank">docker.com/products/docker-desktop</a></li>
            <li>Download the version for your OS (Mac, Windows, or Linux)</li>
            <li>Run the installer and follow the prompts</li>
            <li>Restart your computer when prompted</li>
        </ol>
        
        <h3>Step 2: Verify Installation</h3>
        <p>Open a terminal and run:</p>
        
        <pre><code># Check Docker is installed
$ docker --version
Docker version 24.0.7, build afdd53b

# Check Docker is running  
$ docker run hello-world

Hello from Docker!
This message shows that your installation appears to be working correctly.</code></pre>
        
        <div class="callout callout-warning">
            <strong>Common Issue:</strong> If you get "Cannot connect to the Docker daemon", make sure Docker Desktop is running (check your system tray/menu bar for the whale icon).
        </div>
        
        <h3>Step 3: Test with a Real Container</h3>
        <pre><code># Run an nginx web server
$ docker run -d -p 8080:80 nginx

# This:
# -d          = run in background (detached)
# -p 8080:80  = map your port 8080 to container's port 80
# nginx       = the image to run

# Now open http://localhost:8080 in your browser!

# See running containers
$ docker ps
CONTAINER ID   IMAGE   STATUS         PORTS
abc123...      nginx   Up 2 minutes   0.0.0.0:8080->80/tcp

# Stop the container
$ docker stop abc123</code></pre>
    """
    content += build_section("content", "Installing Docker", install, "📥")
    
    # Essential Docker Commands
    commands = """
        <h3>Commands You'll Use Every Day</h3>
        
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Command</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">What it does</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>docker run</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Create and start a container</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>docker ps</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">List running containers (<code>-a</code> for all)</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>docker stop</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Stop a running container</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>docker rm</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Remove a stopped container</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>docker images</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">List downloaded images</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>docker pull</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Download an image from a registry</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>docker build</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Build an image from a Dockerfile</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>docker logs</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">View container output/logs</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>docker exec</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Run a command inside a container</td>
            </tr>
        </table>
        
        <h3>Common Flags</h3>
        <pre><code># -d = detached (background)
docker run -d nginx

# -p = port mapping (host:container)
docker run -p 3000:3000 myapp

# -v = volume mount (persist data)
docker run -v ./data:/app/data myapp

# -e = environment variable
docker run -e DATABASE_URL=postgres://... myapp

# -it = interactive terminal (for debugging)
docker run -it ubuntu bash

# --rm = remove container when it stops
docker run --rm nginx</code></pre>
    """
    content += build_section("content", "Essential Docker Commands", commands, "⌨️")
    
    # Docker Compose
    compose = """
        <h3>The Problem with Multiple Containers</h3>
        <p>Real applications often need multiple services:</p>
        <ul>
            <li>Your app (Node.js)</li>
            <li>A database (PostgreSQL)</li>
            <li>A cache (Redis)</li>
            <li>Maybe a queue (RabbitMQ)</li>
        </ul>
        
        <p>Running each with <code>docker run</code> gets tedious. You'd need to:</p>
        <ul>
            <li>Start each container manually</li>
            <li>Remember all the ports and volume mounts</li>
            <li>Set up networking between containers</li>
            <li>Stop them all individually</li>
        </ul>
        
        <h3>Docker Compose: One File, All Services</h3>
        <p>Docker Compose lets you define all your services in one YAML file:</p>
        
        <pre><code># compose.yaml
services:
  # Your Node.js application
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgres://postgres:secret@db:5432/myapp
    depends_on:
      - db
      - redis
  
  # PostgreSQL database
  db:
    image: postgres:16
    environment:
      - POSTGRES_PASSWORD=secret
      - POSTGRES_DB=myapp
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  # Redis cache
  redis:
    image: redis:alpine

# Named volumes (data persists when containers restart)
volumes:
  postgres_data:</code></pre>
        
        <h3>Compose Commands</h3>
        <pre><code># Start all services (use -d for background)
$ docker compose up -d

# View logs from all services
$ docker compose logs -f

# Stop all services
$ docker compose stop

# Stop AND remove containers, networks
$ docker compose down

# Stop, remove, AND delete volumes (careful!)
$ docker compose down -v

# Rebuild images after code changes
$ docker compose build
$ docker compose up -d --build  # rebuild and start</code></pre>
        
        <div class="callout callout-tip">
            <strong>Why this matters for RAG:</strong> Our RAG application will have: Node.js app, PostgreSQL with pgvector, and possibly Redis for caching. Docker Compose makes this one-command simple.
        </div>
    """
    content += build_section("content", "Docker Compose", compose, "🎼")
    
    # Hands-on Exercise
    hands_on = """
        <h3>Exercise: Build Your Dev Environment</h3>
        <p>Let's set up the foundation for our RAG project:</p>
        
        <h4>Step 1: Create Project Structure</h4>
        <pre><code>mkdir rag-project
cd rag-project
mkdir -p src/api src/services
touch compose.yaml Dockerfile .env.example</code></pre>
        
        <h4>Step 2: Create the Dockerfile</h4>
        <pre><code># Dockerfile
FROM node:20-alpine

WORKDIR /app

# Copy package files first (for better caching)
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Development command
CMD ["npm", "run", "dev"]</code></pre>
        
        <h4>Step 3: Create compose.yaml</h4>
        <pre><code># compose.yaml
services:
  app:
    build: .
    ports:
      - "3000:3000"
    volumes:
      # Mount source code for hot-reloading
      - ./src:/app/src
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgres://postgres:postgres@db:5432/ragdb
    depends_on:
      db:
        condition: service_healthy

  db:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=ragdb
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:</code></pre>
        
        <h4>Step 4: Create a Simple package.json</h4>
        <pre><code>{
  "name": "rag-project",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "node --watch src/index.js",
    "start": "node src/index.js"
  }
}</code></pre>
        
        <h4>Step 5: Create src/index.js</h4>
        <pre><code>// src/index.js
console.log('🚀 RAG Project Starting...');
console.log('Database URL:', process.env.DATABASE_URL);
console.log('');
console.log('This will become our RAG application!');

// Keep the process running
setInterval(() => {}, 1000);</code></pre>
        
        <h4>Step 6: Run It!</h4>
        <pre><code># Start everything
$ docker compose up -d

# Check status
$ docker compose ps

# View logs
$ docker compose logs -f app

# You should see:
# 🚀 RAG Project Starting...
# Database URL: postgres://postgres:postgres@db:5432/ragdb</code></pre>
        
        <div class="callout callout-tip">
            <strong>Congratulations!</strong> You now have PostgreSQL with pgvector running in a container, ready for our RAG application.
        </div>
    """
    content += build_section("content", "Hands-On: Build Your Environment", hands_on, "💻")
    
    # Gotchas
    gotchas = """
        <h3>Things That Trip People Up</h3>
        
        <div class="callout callout-warning">
            <strong>🚨 Data Loss:</strong> If you don't use volumes, container data is GONE when the container is removed. Always mount volumes for databases!
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 Port Conflicts:</strong> If port 5432 is already in use (local PostgreSQL running?), change the port mapping: <code>5433:5432</code>
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 Container Names:</strong> Container names must be unique. Use <code>docker compose down</code> to clean up old containers before restarting.
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 Apple Silicon (M1/M2):</strong> Most images work fine, but some may need <code>platform: linux/amd64</code> in compose.yaml.
        </div>
        
        <h3>Debugging Tips</h3>
        <pre><code># Container won't start? Check logs:
docker compose logs app

# Need to get inside a container?
docker compose exec app sh

# Database not connecting? Check if it's healthy:
docker compose ps

# Start fresh (nuclear option):
docker compose down -v
docker compose up -d --build</code></pre>
    """
    content += build_section("gotchas", "Common Gotchas & Debugging", gotchas, "⚠️")
    
    # Quiz
    quiz = """
        <div class="quiz-question">
            <p>1. What is the main difference between a Docker image and a container?</p>
            <ul class="quiz-options">
                <li>A) Images run on Linux, containers run on Windows</li>
                <li>B) An image is a template, a container is a running instance</li>
                <li>C) Images are faster than containers</li>
                <li>D) There is no difference</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>2. What does the `-v ./data:/app/data` flag do?</p>
            <ul class="quiz-options">
                <li>A) Makes the container verbose</li>
                <li>B) Mounts a volume from host to container</li>
                <li>C) Sets the version number</li>
                <li>D) Validates the container</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>3. Why use Docker Compose instead of multiple `docker run` commands?</p>
            <ul class="quiz-options">
                <li>A) Compose is faster</li>
                <li>B) Compose manages multiple services, networking, and volumes in one file</li>
                <li>C) Compose uses less memory</li>
                <li>D) docker run is deprecated</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>4. What happens to database data if you run `docker compose down -v`?</p>
            <ul class="quiz-options">
                <li>A) It's backed up automatically</li>
                <li>B) Nothing, volumes are preserved</li>
                <li>C) It's deleted because -v removes volumes</li>
                <li>D) It's moved to a backup location</li>
            </ul>
        </div>
        
        <p style="margin-top: 1rem; color: #8b949e;"><em>Answers: 1-B, 2-B, 3-B, 4-C</em></p>
    """
    content += build_section("quiz", "Check Your Understanding", quiz, "✅")
    
    # References
    refs = """
        <ul>
            <li>
                <a href="https://docs.docker.com/get-started/overview/" target="_blank">
                    Docker Overview - Official Documentation
                </a>
                <span style="color: #8b949e;"> - Comprehensive introduction to Docker</span>
            </li>
            <li>
                <a href="https://docs.docker.com/compose/gettingstarted/" target="_blank">
                    Docker Compose Getting Started
                </a>
                <span style="color: #8b949e;"> - Step-by-step Compose tutorial</span>
            </li>
            <li>
                <a href="https://docs.docker.com/reference/dockerfile/" target="_blank">
                    Dockerfile Reference
                </a>
                <span style="color: #8b949e;"> - All Dockerfile instructions explained</span>
            </li>
            <li>
                <a href="https://hub.docker.com/" target="_blank">
                    Docker Hub
                </a>
                <span style="color: #8b949e;"> - Public registry of Docker images</span>
            </li>
            <li>
                <a href="https://github.com/pgvector/pgvector" target="_blank">
                    pgvector on GitHub
                </a>
                <span style="color: #8b949e;"> - The PostgreSQL vector extension we'll use</span>
            </li>
        </ul>
    """
    content += build_section("references", "Sources & Further Reading", refs, "📚")
    
    # Generate HTML
    html = MODULE_TEMPLATE.format(
        title="Module 1: Docker & Development Environment",
        description="Master Docker and set up your complete development environment for building RAG applications",
        time_estimate="~60 minutes",
        difficulty="Beginner",
        module_number="1",
        content=content,
        prev_link='<a href="../module-0-overview/index.html">← Previous: The Big Picture</a>',
        next_link='<a href="../module-2-postgres/index.html">Next: Database Foundations →</a>'
    )
    
    # Save
    output_dir = os.path.join(COURSE_DIR, "module-1-docker")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
    
    print("✅ Generated Module 1: Docker & Development Environment")


def build_module_2():
    """Build Module 2: Database Foundations"""
    
    content = ""
    
    # Objectives
    objectives = """
        <ul>
            <li>Understand what databases are and why PostgreSQL</li>
            <li>Master essential SQL: SELECT, INSERT, UPDATE, DELETE, JOIN</li>
            <li>Understand data types, constraints, and indexes</li>
            <li>Learn about pgvector extension for vector storage</li>
            <li>Connect to PostgreSQL from Node.js</li>
        </ul>
    """
    content += build_section("objectives", "What You'll Learn", objectives, "🎯")
    
    # Prerequisites
    prereqs = """
        <p>Before starting this module, you should have:</p>
        <ul>
            <li>Completed Module 1 (Docker running)</li>
            <li>The RAG project from Module 1 with PostgreSQL container</li>
            <li>A code editor (VS Code recommended)</li>
        </ul>
        <div class="callout callout-info">
            <strong>No SQL experience needed.</strong> We only use a small, practical subset of SQL in this course (create tables, insert rows, and run a few queries). If you know some SQL already, the later sections on pgvector will be new.
        </div>
    """
    content += build_section("prerequisites", "Prerequisites", prereqs, "📋")
    
    # What is a Database
    db_intro = """
        <h3>Why Do We Need Databases?</h3>
        <p>Imagine you're building an app that stores user profiles. You could save them to a file:</p>
        
        <pre><code>// users.json
[
  { "id": 1, "name": "Alice", "email": "alice@example.com" },
  { "id": 2, "name": "Bob", "email": "bob@example.com" }
]</code></pre>
        
        <p>This works for a prototype, but problems emerge quickly:</p>
        <ul>
            <li><strong>Speed:</strong> Finding a user requires reading the entire file</li>
            <li><strong>Concurrency:</strong> What if two requests try to update at the same time?</li>
            <li><strong>Size:</strong> What when you have millions of users?</li>
            <li><strong>Relationships:</strong> How do you link users to their orders?</li>
            <li><strong>Data integrity:</strong> How do you ensure emails are unique?</li>
        </ul>
        
        <p><strong>Databases solve all of these problems</strong> with specialized data structures, indexing, locking, and query optimization.</p>
        
        <h3>Why PostgreSQL?</h3>
        <p>There are many databases, but PostgreSQL stands out:</p>
        
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Feature</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Why It Matters for RAG</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">pgvector extension</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Store and search embeddings directly in your DB</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">JSONB support</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Store document metadata flexibly</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Full-text search</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Combine vector search with keyword search</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Reliability</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">ACID compliance, data won't get corrupted</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Free & Open Source</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">No licensing fees, huge community</td>
            </tr>
        </table>
        
        <div class="callout callout-tip">
            <strong>Key insight:</strong> By using pgvector, we can store documents AND their embeddings in the same database. No need for a separate vector database like Pinecone or Weaviate (though those are great too).
        </div>
    """
    content += build_section("content", "What is a Database?", db_intro, "🗄️")
    
    # SQL Basics
    sql_basics = """
        <h3>Connecting to Your Database</h3>
        <p>First, let's connect to the PostgreSQL container from Module 1:</p>
        
        <pre><code># Make sure your containers are running
$ docker compose up -d

# Connect to PostgreSQL inside the container
$ docker compose exec db psql -U postgres -d ragdb

# You should see:
ragdb=# </code></pre>
        
        <p>You're now in the PostgreSQL interactive terminal. Let's learn SQL!</p>
        
        <h3>CREATE TABLE - Defining Structure</h3>
        <pre><code>-- Create a table for documents
CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,     -- Auto-incrementing ID
    title       TEXT NOT NULL,          -- Required text
    content     TEXT NOT NULL,          -- The document text
    created_at  TIMESTAMP DEFAULT NOW() -- Auto-set timestamp
);

-- Verify it was created
\\dt</code></pre>
        
        <div class="callout callout-info">
            <strong>SQL Keywords:</strong> SQL is case-insensitive, but convention is UPPERCASE for keywords (SELECT, FROM) and lowercase for your names (documents, title).
        </div>
        
        <h3>INSERT - Adding Data</h3>
        <pre><code>-- Insert a single document
INSERT INTO documents (title, content) 
VALUES ('Getting Started', 'Welcome to our documentation...');

-- Insert multiple documents
INSERT INTO documents (title, content) VALUES
    ('Installation', 'First, install Node.js from nodejs.org...'),
    ('Configuration', 'Create a .env file with your settings...'),
    ('API Reference', 'The API exposes the following endpoints...');</code></pre>
        
        <h3>SELECT - Reading Data</h3>
        <pre><code>-- Get all documents
SELECT * FROM documents;

-- Get specific columns
SELECT id, title FROM documents;

-- Filter with WHERE
SELECT * FROM documents WHERE id = 1;

-- Search text with LIKE
SELECT * FROM documents WHERE title LIKE '%API%';

-- Case-insensitive search with ILIKE
SELECT * FROM documents WHERE content ILIKE '%install%';

-- Order results
SELECT * FROM documents ORDER BY created_at DESC;

-- Limit results
SELECT * FROM documents ORDER BY created_at DESC LIMIT 5;</code></pre>
        
        <h3>UPDATE - Modifying Data</h3>
        <pre><code>-- Update a single document
UPDATE documents 
SET content = 'Updated content here...' 
WHERE id = 1;

-- Update multiple fields
UPDATE documents 
SET title = 'New Title', content = 'New content' 
WHERE id = 1;

-- ⚠️ Without WHERE, updates ALL rows!
-- UPDATE documents SET title = 'Oops'; -- Don't do this!</code></pre>
        
        <h3>DELETE - Removing Data</h3>
        <pre><code>-- Delete specific document
DELETE FROM documents WHERE id = 1;

-- Delete with condition
DELETE FROM documents WHERE created_at < '2024-01-01';

-- ⚠️ Without WHERE, deletes ALL rows!
-- DELETE FROM documents; -- Be very careful!</code></pre>
    """
    content += build_section("content", "SQL Fundamentals", sql_basics, "📝")

    sql_cheatsheet = """
        <h3>The Only SQL You Need (Cheat Sheet)</h3>
        <p>If you're new to SQL, don't try to learn everything at once. For this course, these are the core commands you'll reuse again and again:</p>

        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">SQL</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Meaning</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Why it matters for RAG</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>CREATE TABLE</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Define storage</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Store documents + chunks + embeddings</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>INSERT</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Save rows</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Ingestion writes chunks + embeddings</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>SELECT</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Read rows</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Retrieval reads top matching chunks</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>WHERE</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Filter rows</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Filter by document, tenant, permissions</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>ORDER BY ... LIMIT</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Top results</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">“Top-K” retrieval is literally this</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>JOIN</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Combine tables</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Join chunks back to document titles/metadata</td>
            </tr>
        </table>

        <h3>RAG Retrieval Query (Template)</h3>
        <p>This is the most important query in a RAG app. You can treat it like a reusable template:</p>
        <pre><code>SELECT
  c.content,
  d.title,
  1 - (c.embedding <=> $1::vector) AS similarity
FROM chunks c
JOIN documents d ON c.document_id = d.id
ORDER BY c.embedding <=> $1::vector
LIMIT 5;</code></pre>

        <div class="callout callout-tip">
            <strong>Key idea:</strong> You don't need “deep SQL” to build RAG. You need one solid retrieval query + clean ingestion.
        </div>
    """
    content += build_section("content", "SQL Cheat Sheet (For This Course)", sql_cheatsheet, "🧾")

    sql_minilab = """
        <h3>Mini-Lab: SQL Confidence in 15 Minutes</h3>
        <p>This is a guided practice. Copy/paste each block into <code>psql</code> and compare with the expected output. If you complete this, you have enough SQL for the entire course.</p>

        <h4>Step 0: Connect to Postgres</h4>
        <pre><code># Start containers (if not running)
    $ docker compose up -d

    # Open Postgres shell
    $ docker compose exec db psql -U postgres -d ragdb</code></pre>

        <div class="callout callout-info">
            <strong>psql tip:</strong> Use <code>\\dt</code> to list tables, <code>\\d table_name</code> to describe a table, and <code>\\q</code> to quit.
        </div>

        <h4>Exercise 1: Create a table</h4>
        <pre><code>CREATE TABLE IF NOT EXISTS mini_docs (
      id SERIAL PRIMARY KEY,
      title TEXT NOT NULL,
      content TEXT NOT NULL,
      created_at TIMESTAMP DEFAULT NOW()
    );

    \\dt</code></pre>
        <p><em>Expected:</em> You should see <code>mini_docs</code> in the table list.</p>

        <h4>Exercise 2: Insert a few rows</h4>
        <pre><code>INSERT INTO mini_docs (title, content) VALUES
      ('Install', 'Install Node.js and Docker.'),
      ('Auth', 'Use API keys and rotate them.'),
      ('Errors', 'Handle timeouts and retries.');

    SELECT id, title FROM mini_docs ORDER BY id;</code></pre>
        <p><em>Expected:</em> 3 rows with IDs 1..3 (or continuing IDs if you ran before).</p>

        <h4>Exercise 3: Filter results</h4>
        <pre><code>SELECT id, title
    FROM mini_docs
    WHERE title ILIKE '%auth%';</code></pre>
        <p><em>Expected:</em> 1 row (Auth).</p>

        <h4>Exercise 4: Update safely (with WHERE)</h4>
        <pre><code>UPDATE mini_docs
    SET content = 'Use API keys, rate limits, and rotate secrets.'
    WHERE title = 'Auth';

    SELECT title, content FROM mini_docs WHERE title = 'Auth';</code></pre>
        <p><em>Expected:</em> Updated content for Auth only.</p>

        <h4>Exercise 5: Join (the pattern you’ll reuse in RAG)</h4>
        <p>RAG commonly stores documents in one table and chunks in another table. This join pattern lets you retrieve chunks and still show the document title.</p>
        <pre><code>CREATE TABLE IF NOT EXISTS mini_chunks (
      id SERIAL PRIMARY KEY,
      doc_id INTEGER NOT NULL REFERENCES mini_docs(id),
      chunk_text TEXT NOT NULL
    );

    INSERT INTO mini_chunks (doc_id, chunk_text) VALUES
      (1, 'Docker installs on Linux/Windows/Mac.'),
      (1, 'Use docker compose for local dev.'),
      (2, 'Never expose keys in frontend code.');

    SELECT d.title, c.chunk_text
    FROM mini_chunks c
    JOIN mini_docs d ON c.doc_id = d.id
    ORDER BY d.id, c.id;</code></pre>

        <div class="callout callout-tip">
            <strong>Big takeaway:</strong> That <code>JOIN</code> is exactly what you’ll do in RAG: search chunks → join back to document metadata.
        </div>
        """

    content += build_section("content", "Mini-Lab: Practice SQL", sql_minilab, "🧪")

    troubleshooting = """
            <h3>Common Errors & Quick Fixes</h3>
            <p>If you get stuck in this course, it’s usually one of these. Start here before you assume you “don’t know SQL”.</p>

            <h4>Docker / Compose issues</h4>

            <h5>1) <code>docker: command not found</code></h5>
            <p><em>Fix:</em> Install Docker Desktop (Mac/Windows) or Docker Engine (Linux). Then re-open your terminal.</p>

            <h5>2) <code>docker compose</code> fails (older installs)</h5>
            <p><em>Fix:</em> Try <code>docker-compose</code> instead of <code>docker compose</code>, or upgrade Docker.</p>

            <h5>3) Port already in use (e.g. 5432)</h5>
            <p><em>Symptoms:</em> compose fails to start, or Postgres container keeps restarting.</p>
            <p><em>Fix options:</em></p>
            <ul>
                <li>Stop local Postgres: <code>sudo systemctl stop postgresql</code> (Linux, if applicable)</li>
                <li>Or change the host port in <code>docker-compose.yml</code> (e.g. <code>15432:5432</code>)</li>
            </ul>

            <h4>Postgres connection issues</h4>

            <h5>4) <code>could not connect to server</code> / connection refused</h5>
            <p><em>Fix:</em> Confirm the container is running and healthy:</p>
            <pre><code>docker compose ps
    docker compose logs db --tail 100</code></pre>
            <p>Then try connecting again:</p>
            <pre><code>docker compose exec db psql -U postgres -d ragdb</code></pre>

            <h5>5) <code>database "ragdb" does not exist</code></h5>
            <p><em>Fix:</em> Either create it or connect to <code>postgres</code> first:</p>
            <pre><code>docker compose exec db psql -U postgres -d postgres

    CREATE DATABASE ragdb;
            \\q

    docker compose exec db psql -U postgres -d ragdb</code></pre>

            <h5>6) <code>role "postgres" does not exist</code></h5>
            <p><em>Fix:</em> Your compose config may use a different user. Check your environment variables in compose, then connect with that user. If you’re unsure:</p>
            <pre><code>docker compose exec db psql -U postgres -d postgres</code></pre>
            <p>If that fails, open <code>docker-compose.yml</code> and look for <code>POSTGRES_USER</code>.</p>

            <h4>pgvector issues</h4>

            <h5>7) <code>type "vector" does not exist</code></h5>
            <p><em>Fix:</em> Enable the extension in your database:</p>
            <pre><code>CREATE EXTENSION IF NOT EXISTS vector;</code></pre>

            <h5>8) <code>extension "vector" is not available</code></h5>
            <p><em>Fix:</em> Your Postgres image doesn’t include pgvector. Use an image that includes it (or install it). In this course, the provided Docker setup should already handle this—so if you see this, double-check you’re using the course’s <code>docker-compose.yml</code>.</p>

            <h4>RAG retrieval query issues</h4>
            <h5>9) <code>operator does not exist: vector &lt;=&gt; vector</code></h5>
            <p><em>Fix:</em> Same root cause as above: pgvector isn’t enabled/installed in the DB you’re connected to.</p>

            <h5>10) Dimension mismatch when inserting embeddings</h5>
            <p><em>Symptoms:</em> errors mentioning vector length/dimensions.</p>
            <p><em>Fix:</em> Ensure your column is declared with the correct dimension, e.g. <code>VECTOR(1536)</code> for many embedding models, and that your embedding generator uses the same model consistently.</p>

            <div class="callout callout-tip">
                <strong>Debugging mindset:</strong> In GenAI apps, 80% of “it’s broken” is just environment + connectivity. Check containers, logs, and DB extensions first.
            </div>
        """

    content += build_section("content", "Troubleshooting", troubleshooting, "🧰")
    
    # Data Types and Constraints
    types_section = """
        <h3>Common PostgreSQL Data Types</h3>
        
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Type</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Use For</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Example</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>TEXT</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Variable-length strings</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Document content</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>VARCHAR(n)</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Limited-length strings</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Usernames (max 50 chars)</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>INTEGER</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Whole numbers</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Counts, IDs</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>FLOAT/REAL</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Decimal numbers</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Similarity scores</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>BOOLEAN</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">True/False</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">is_processed flag</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>TIMESTAMP</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Date and time</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">created_at</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>JSONB</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Flexible JSON data</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Document metadata</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>VECTOR(n)</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Embedding vectors (pgvector)</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Document embeddings</td>
            </tr>
        </table>
        
        <h3>Constraints - Data Integrity Rules</h3>
        <pre><code>CREATE TABLE users (
    id          SERIAL PRIMARY KEY,          -- Unique identifier
    email       TEXT NOT NULL UNIQUE,        -- Required & unique
    name        TEXT NOT NULL,               -- Required
    role        TEXT DEFAULT 'user',         -- Default value
    created_at  TIMESTAMP DEFAULT NOW()
);

-- What happens when you violate constraints:
INSERT INTO users (email, name) VALUES ('alice@example.com', 'Alice');  -- ✅ Works
INSERT INTO users (email, name) VALUES ('alice@example.com', 'Alice2'); -- ❌ Error: duplicate email
INSERT INTO users (name) VALUES ('Bob');                                -- ❌ Error: email is required</code></pre>
    """
    content += build_section("content", "Data Types & Constraints", types_section, "🔧")
    
    # pgvector Section
    pgvector_section = """
        <h3>What is pgvector?</h3>
        <p>pgvector is a PostgreSQL extension that adds a <code>vector</code> data type and operators for similarity search. This is the key technology that makes our RAG application possible!</p>
        
        <blockquote style="border-left: 4px solid #818cf8; padding-left: 1rem; color: #c9d1d9; margin: 1rem 0;">
            "Vectors in Supabase are enabled via pgvector, a Postgres extension for storing and querying vectors in Postgres. It can be used to store embeddings."
            <footer style="color: #8b949e; margin-top: 0.5rem;">— Supabase Documentation</footer>
        </blockquote>
        
        <h3>Enable pgvector</h3>
        <pre><code>-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify it's enabled
\\dx</code></pre>
        
        <h3>Create a Table with Vectors</h3>
        <pre><code>-- Create table for documents with embeddings
CREATE TABLE document_chunks (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    content     TEXT NOT NULL,
    embedding   VECTOR(1536),        -- OpenAI embeddings are 1536 dimensions
    metadata    JSONB DEFAULT '{}'
);

-- The number in VECTOR(n) is the dimension count:
-- - OpenAI text-embedding-ada-002: 1536
-- - OpenAI text-embedding-3-small: 1536  
-- - OpenAI text-embedding-3-large: 3072
-- - Sentence Transformers (many): 384-768</code></pre>
        
        <div class="callout callout-warning">
            <strong>Important:</strong> The vector dimensions must match your embedding model. If you use OpenAI's text-embedding-ada-002, use VECTOR(1536). Using the wrong dimension will cause errors!
        </div>
        
        <h3>Insert Vectors</h3>
        <pre><code>-- Insert a vector (normally you'd get this from an embedding API)
INSERT INTO document_chunks (document_id, content, embedding) 
VALUES (
    1,
    'This is a chunk of text from document 1',
    '[0.1, 0.2, 0.3, ...]'::vector  -- 1536 float values
);</code></pre>
        
        <h3>Vector Similarity Search</h3>
        <p>This is the magic of RAG! pgvector supports 3 distance operators:</p>
        
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Operator</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Distance Type</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Best For</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>&lt;-&gt;</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Euclidean (L2)</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">General purpose</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>&lt;#&gt;</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Inner Product (negative)</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Maximum inner product</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>&lt;=&gt;</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Cosine</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Normalized vectors (most common)</td>
            </tr>
        </table>
        
        <pre><code>-- Find the 5 most similar chunks to a query embedding
SELECT 
    id,
    content,
    1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM document_chunks
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;</code></pre>
        
        <div class="callout callout-tip">
            <strong>Cosine Similarity:</strong> The <code>&lt;=&gt;</code> operator returns cosine <em>distance</em> (0 = identical, 2 = opposite). To get similarity (1 = identical, 0 = unrelated), use <code>1 - (embedding <=> query)</code>.
        </div>
        
        <h3>Creating Vector Indexes</h3>
        <p>Without an index, searches scan every row. With millions of embeddings, this becomes slow. pgvector supports special vector indexes:</p>
        
        <pre><code>-- IVFFlat index (faster, less accurate)
CREATE INDEX ON document_chunks 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- HNSW index (slower to build, more accurate)
CREATE INDEX ON document_chunks 
USING hnsw (embedding vector_cosine_ops);</code></pre>
        
        <div class="callout callout-info">
            <strong>When to add indexes:</strong> Start without indexes while developing. Once you have 10,000+ vectors, add an HNSW index for better performance.
        </div>
    """
    content += build_section("content", "pgvector - Vector Storage", pgvector_section, "🔢")
    
    # Node.js Connection
    nodejs_section = """
        <h3>Connecting from Node.js</h3>
        <p>Let's connect our Node.js app to PostgreSQL. We'll use the <code>pg</code> package:</p>
        
        <pre><code># Install the pg package
npm install pg</code></pre>
        
        <h4>Create a Database Client</h4>
        <pre><code>// src/db.js
import pg from 'pg';
const { Pool } = pg;

// Create a connection pool
const pool = new Pool({
    connectionString: process.env.DATABASE_URL
    // For local Docker: postgres://postgres:postgres@db:5432/ragdb
});

// Test the connection
pool.query('SELECT NOW()')
    .then(res => console.log('✅ Database connected:', res.rows[0].now))
    .catch(err => console.error('❌ Database error:', err));

export default pool;</code></pre>
        
        <h4>Query Examples</h4>
        <pre><code>// src/documents.js
import pool from './db.js';

// Get all documents
export async function getAllDocuments() {
    const result = await pool.query('SELECT * FROM documents ORDER BY created_at DESC');
    return result.rows;
}

// Get document by ID
export async function getDocument(id) {
    const result = await pool.query('SELECT * FROM documents WHERE id = $1', [id]);
    return result.rows[0];
}

// Create document
export async function createDocument(title, content) {
    const result = await pool.query(
        'INSERT INTO documents (title, content) VALUES ($1, $2) RETURNING *',
        [title, content]
    );
    return result.rows[0];
}

// Search by similarity (we'll expand this in Module 5)
export async function searchSimilar(embedding, limit = 5) {
    const result = await pool.query(`
        SELECT id, content, 1 - (embedding <=> $1) as similarity
        FROM document_chunks
        ORDER BY embedding <=> $1
        LIMIT $2
    `, [JSON.stringify(embedding), limit]);
    return result.rows;
}</code></pre>
        
        <div class="callout callout-warning">
            <strong>SQL Injection Warning:</strong> NEVER build queries with string concatenation! Always use parameterized queries (<code>$1</code>, <code>$2</code>) like the examples above. This protects against SQL injection attacks.
        </div>
    """
    content += build_section("content", "Node.js & PostgreSQL", nodejs_section, "🟢")
    
    # Hands-on Exercise
    hands_on = """
        <h3>Exercise: Set Up Your RAG Database Schema</h3>
        <p>Let's create the database schema we'll use for our RAG application:</p>
        
        <h4>Step 1: Connect to PostgreSQL</h4>
        <pre><code>$ docker compose exec db psql -U postgres -d ragdb</code></pre>
        
        <h4>Step 2: Create the Schema</h4>
        <pre><code>-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Sources table (where documents come from)
CREATE TABLE sources (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,  -- 'file', 'url', 'api'
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Documents table (the actual documents)
CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    source_id   INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Chunks table (split documents with embeddings)
CREATE TABLE chunks (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    content     TEXT NOT NULL,
    embedding   VECTOR(1536),  -- OpenAI dimensions
    token_count INTEGER,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Create index for similarity search
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);</code></pre>
        
        <h4>Step 3: Add Test Data</h4>
        <pre><code>-- Insert a source
INSERT INTO sources (name, type) VALUES ('Manual Entry', 'manual');

-- Insert a document
INSERT INTO documents (source_id, title, content) VALUES 
    (1, 'RAG Introduction', 'RAG stands for Retrieval-Augmented Generation...');

-- Verify
SELECT s.name as source, d.title, d.content 
FROM documents d 
JOIN sources s ON d.source_id = s.id;</code></pre>
        
        <h4>Step 4: Exit PostgreSQL</h4>
        <pre><code>-- Exit psql
\\q</code></pre>
        
        <div class="callout callout-tip">
            <strong>Congratulations!</strong> You now have a database schema ready for your RAG application. In Module 5, we'll populate the chunks table with real embeddings.
        </div>
    """
    content += build_section("content", "Hands-On: Create RAG Schema", hands_on, "💻")
    
    # Gotchas
    gotchas = """
        <h3>Common Database Mistakes</h3>
        
        <div class="callout callout-warning">
            <strong>🚨 UPDATE/DELETE without WHERE:</strong> Always double-check your WHERE clause before running UPDATE or DELETE. Without it, you'll modify ALL rows!
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 Wrong vector dimensions:</strong> If you get "different vector dimensions" errors, check that your embedding model dimensions match your VECTOR(n) column definition.
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 Connection leaks:</strong> Always use connection pooling. Don't create new connections for each query - use a Pool like our example.
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 N+1 queries:</strong> Avoid querying in loops. Use JOINs or fetch all data at once instead.
        </div>
        
        <h3>Useful psql Commands</h3>
        <pre><code>\\dt          -- List tables
\\d tablename -- Describe table structure
\\dx          -- List extensions
\\conninfo    -- Show connection info
\\q           -- Quit
\\?           -- Help</code></pre>
    """
    content += build_section("gotchas", "Common Gotchas & Tips", gotchas, "⚠️")
    
    # Quiz
    quiz = """
        <div class="quiz-question">
            <p>1. Which SQL command adds new rows to a table?</p>
            <ul class="quiz-options">
                <li>A) UPDATE</li>
                <li>B) INSERT</li>
                <li>C) CREATE</li>
                <li>D) ADD</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>2. What does the <code>&lt;=&gt;</code> operator do in pgvector?</p>
            <ul class="quiz-options">
                <li>A) Checks if two vectors are equal</li>
                <li>B) Calculates Euclidean distance</li>
                <li>C) Calculates cosine distance</li>
                <li>D) Adds two vectors together</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>3. Why should you use parameterized queries ($1, $2) instead of string concatenation?</p>
            <ul class="quiz-options">
                <li>A) They're faster</li>
                <li>B) They prevent SQL injection attacks</li>
                <li>C) They use less memory</li>
                <li>D) PostgreSQL requires them</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>4. What vector dimension should you use for OpenAI's text-embedding-ada-002 model?</p>
            <ul class="quiz-options">
                <li>A) 384</li>
                <li>B) 768</li>
                <li>C) 1536</li>
                <li>D) 3072</li>
            </ul>
        </div>
        
        <p style="margin-top: 1rem; color: #8b949e;"><em>Answers: 1-B, 2-C, 3-B, 4-C</em></p>
    """
    content += build_section("quiz", "Check Your Understanding", quiz, "✅")
    
    # References
    refs = """
        <ul>
            <li>
                <a href="https://www.postgresql.org/docs/current/tutorial.html" target="_blank">
                    PostgreSQL Tutorial - Official Docs
                </a>
                <span style="color: #8b949e;"> - Complete PostgreSQL introduction</span>
            </li>
            <li>
                <a href="https://github.com/pgvector/pgvector" target="_blank">
                    pgvector on GitHub
                </a>
                <span style="color: #8b949e;"> - Official pgvector extension</span>
            </li>
            <li>
                <a href="https://supabase.com/docs/guides/ai/vector-columns" target="_blank">
                    Supabase Vector Columns Guide
                </a>
                <span style="color: #8b949e;"> - Excellent pgvector tutorial</span>
            </li>
            <li>
                <a href="https://node-postgres.com/" target="_blank">
                    node-postgres (pg)
                </a>
                <span style="color: #8b949e;"> - Node.js PostgreSQL client</span>
            </li>
            <li>
                <a href="https://www.prisma.io/docs/concepts/database-connectors/postgresql" target="_blank">
                    Prisma with PostgreSQL
                </a>
                <span style="color: #8b949e;"> - Popular ORM alternative</span>
            </li>
        </ul>
    """
    content += build_section("references", "Sources & Further Reading", refs, "📚")
    
    # Generate HTML
    html = MODULE_TEMPLATE.format(
        title="Module 2: Database Foundations",
        description="Master PostgreSQL, SQL essentials, and pgvector for storing embeddings",
        time_estimate="~90 minutes",
        difficulty="Beginner",
        module_number="2",
        content=content,
        prev_link='<a href="../module-1-docker/index.html">← Previous: Docker Environment</a>',
        next_link='<a href="../module-3-llm/index.html">Next: LLM Fundamentals →</a>'
    )
    
    # Save
    output_dir = os.path.join(COURSE_DIR, "module-2-postgres")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
    
    print("✅ Generated Module 2: Database Foundations")


def build_module_3():
    """Build Module 3: LLM Fundamentals"""
    
    content = ""
    
    # Objectives
    objectives = """
        <ul>
            <li>Understand what LLMs are and how they work (at a high level)</li>
            <li>Master the OpenAI API for chat completions</li>
            <li>Learn prompt engineering fundamentals</li>
            <li>Understand tokens, context windows, and model parameters</li>
            <li>Build a simple chatbot with Node.js</li>
        </ul>
    """
    content += build_section("objectives", "What You'll Learn", objectives, "🎯")
    
    # Prerequisites
    prereqs = """
        <p>Before starting this module, you should have:</p>
        <ul>
            <li>Completed Modules 1-2 (Docker + Database running)</li>
            <li>An OpenAI API key (get one at <a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com</a>)</li>
            <li>Basic JavaScript/TypeScript knowledge</li>
        </ul>
        <div class="callout callout-warning">
            <strong>API Costs:</strong> OpenAI charges for API usage. For this course, expect to spend $1-5 total. Set a spending limit in your OpenAI dashboard!
        </div>
    """
    content += build_section("prerequisites", "Prerequisites", prereqs, "📋")
    
    # What are LLMs
    llm_intro = """
        <h3>What is a Large Language Model?</h3>
        <p>A Large Language Model (LLM) is a neural network trained on massive amounts of text to predict the next word in a sequence. But this simple objective leads to remarkable capabilities:</p>
        
        <ul>
            <li><strong>Understanding:</strong> Grasping context, nuance, and intent</li>
            <li><strong>Generation:</strong> Writing coherent, contextually appropriate text</li>
            <li><strong>Reasoning:</strong> Following logical chains of thought</li>
            <li><strong>Translation:</strong> Converting between languages and formats</li>
        </ul>
        
        <h3>How LLMs Work (Simplified)</h3>
        <div style="background: #161b22; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
            <p style="font-family: monospace; margin: 0;">
                Input: "The capital of France is"<br><br>
                Model predicts probabilities:<br>
                → "Paris" (92%)<br>
                → "located" (3%)<br>
                → "a" (2%)<br>
                → ... thousands more options<br><br>
                Output: "Paris"
            </p>
        </div>
        
        <p>The model does this one token at a time, using all previous tokens as context. This is why LLMs are called "autoregressive" - each prediction depends on all previous predictions.</p>
        
        <h3>Key Terminology</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Term</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Meaning</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><strong>Token</strong></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">A piece of text (roughly 4 characters or 3/4 of a word)</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><strong>Context Window</strong></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Maximum tokens the model can "see" at once (e.g., 128K for GPT-4)</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><strong>Temperature</strong></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Randomness control (0 = deterministic, 1 = creative)</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><strong>Prompt</strong></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">The input text you send to the model</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><strong>Completion</strong></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">The output text the model generates</td>
            </tr>
        </table>
        
        <div class="callout callout-tip">
            <strong>Token Counting:</strong> A rough rule: 1 token ≈ 4 characters in English. "Hello world" is 2 tokens. Use OpenAI's <a href="https://platform.openai.com/tokenizer" target="_blank">tokenizer tool</a> to count exactly.
        </div>
    """
    content += build_section("content", "Understanding LLMs", llm_intro, "🧠")
    
    # OpenAI API
    openai_section = """
        <h3>Setting Up OpenAI</h3>
        <p>First, let's install the OpenAI SDK and set up authentication:</p>
        
        <pre><code># Install the OpenAI package
npm install openai

# Add your API key to .env
echo "OPENAI_API_KEY=sk-your-key-here" >> .env</code></pre>
        
        <h4>Basic API Call</h4>
        <pre><code>// src/llm.ts
import OpenAI from 'openai';
import 'dotenv/config';

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

async function chat(userMessage: string): Promise&lt;string&gt; {
    const response = await openai.chat.completions.create({
        model: 'gpt-4o-mini',  // Fast and cheap
        messages: [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: userMessage }
        ]
    });
    
    return response.choices[0].message.content || '';
}

// Test it
const answer = await chat('What is RAG in AI?');
console.log(answer);</code></pre>
        
        <h3>Understanding the Messages Array</h3>
        <p>Chat models use a conversation format with three roles:</p>
        
        <pre><code>const messages = [
    // SYSTEM: Sets the AI's behavior (hidden from user)
    { 
        role: 'system', 
        content: 'You are a helpful coding tutor. Be concise.' 
    },
    
    // USER: What the human says
    { 
        role: 'user', 
        content: 'Explain async/await in JavaScript' 
    },
    
    // ASSISTANT: What the AI says (for conversation history)
    { 
        role: 'assistant', 
        content: 'Async/await is syntax sugar for Promises...' 
    },
    
    // More USER messages continue the conversation
    { 
        role: 'user', 
        content: 'Can you show an example?' 
    }
];</code></pre>
        
        <h3>Model Parameters</h3>
        <pre><code>const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: messages,
    
    // Temperature: 0-2, controls randomness
    // 0 = deterministic, 1 = balanced, 2 = very random
    temperature: 0.7,
    
    // Max tokens to generate in response
    max_tokens: 1000,
    
    // Stop generating at these sequences
    stop: ['\\n\\n', 'END'],
    
    // Return multiple completions
    n: 1
});</code></pre>
        
        <div class="callout callout-info">
            <strong>Model Selection:</strong><br>
            • <code>gpt-4o-mini</code> - Fast, cheap, good for most tasks ($0.15/1M input tokens)<br>
            • <code>gpt-4o</code> - More capable, better reasoning ($2.50/1M input tokens)<br>
            • <code>gpt-4-turbo</code> - Legacy, still good ($10/1M input tokens)
        </div>
    """
    content += build_section("content", "The OpenAI API", openai_section, "🔌")
    
    # Prompt Engineering
    prompting_section = """
        <h3>What is Prompt Engineering?</h3>
        
        <blockquote style="border-left: 4px solid #818cf8; padding-left: 1rem; color: #c9d1d9; margin: 1rem 0;">
            "Prompt engineering is a relatively new discipline for developing and optimizing prompts to efficiently use language models (LMs) for a wide variety of applications and research topics. Prompt engineering skills help to better understand the capabilities and limitations of large language models (LLMs)."
            <footer style="color: #8b949e; margin-top: 0.5rem;">— Prompt Engineering Guide</footer>
        </blockquote>
        
        <h3>Core Prompting Techniques</h3>
        
        <h4>1. Be Specific and Clear</h4>
        <pre><code>// ❌ Vague
"Write about dogs"

// ✅ Specific
"Write a 100-word paragraph about the health benefits of 
walking dogs daily, targeting first-time dog owners."</code></pre>
        
        <h4>2. Provide Context</h4>
        <pre><code>// ❌ No context
"Fix this code: const x = await getData()"

// ✅ With context
"I'm using Node.js with TypeScript. This function runs but 
returns undefined. Here's the full function:
[code]
The getData() function should return a User object."</code></pre>
        
        <h4>3. Use Examples (Few-Shot Prompting)</h4>
        <pre><code>const systemPrompt = `
You classify customer feedback as positive, negative, or neutral.

Examples:
- "Love this product!" → positive
- "Worst purchase ever" → negative  
- "It works as expected" → neutral

Now classify the following:
`;</code></pre>
        
        <h4>4. Chain of Thought (CoT)</h4>
        <pre><code>// Ask the model to think step by step
const prompt = `
Solve this problem step by step:

A store has 50 apples. They sell 23 in the morning and 
receive a shipment of 30 in the afternoon. How many 
apples do they have at the end of the day?

Think through this step by step before giving the final answer.
`;</code></pre>
        
        <h4>5. Role Assignment</h4>
        <pre><code>const systemPrompt = `
You are an expert TypeScript developer with 10 years of experience.
You always:
- Write clean, well-documented code
- Follow best practices
- Explain your reasoning
- Consider edge cases
`;</code></pre>
        
        <h3>The System Prompt Pattern</h3>
        <p>A well-structured system prompt has these components:</p>
        
        <pre><code>const systemPrompt = `
# Role
You are a technical documentation assistant for a Node.js project.

# Context  
You help developers understand and use our RAG API.

# Guidelines
- Be concise but thorough
- Include code examples when relevant
- If unsure, say "I don't know" rather than guessing
- Format responses in Markdown

# Constraints
- Only answer questions about our API
- Don't provide information about competitors
- Keep responses under 500 words unless asked for more
`;</code></pre>
    """
    content += build_section("content", "Prompt Engineering", prompting_section, "✍️")
    
    # Streaming and advanced usage
    advanced_section = """
        <h3>Streaming Responses</h3>
        <p>For better UX, stream responses instead of waiting for the complete answer:</p>
        
        <pre><code>async function streamChat(userMessage: string) {
    const stream = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: userMessage }
        ],
        stream: true  // Enable streaming
    });
    
    // Process chunks as they arrive
    for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || '';
        process.stdout.write(content);  // Print without newline
    }
    console.log();  // Final newline
}

await streamChat('Tell me a short story about a robot.');</code></pre>
        
        <h3>Handling Errors</h3>
        <pre><code>import OpenAI from 'openai';

async function safeChatCall(message: string) {
    try {
        const response = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            messages: [{ role: 'user', content: message }]
        });
        return response.choices[0].message.content;
        
    } catch (error) {
        if (error instanceof OpenAI.APIError) {
            console.error('API Error:', error.status, error.message);
            
            if (error.status === 429) {
                // Rate limited - wait and retry
                console.log('Rate limited, waiting...');
                await new Promise(r => setTimeout(r, 5000));
                return safeChatCall(message);  // Retry
            }
            
            if (error.status === 401) {
                throw new Error('Invalid API key');
            }
        }
        throw error;
    }
}</code></pre>
        
        <h3>Token Counting</h3>
        <p>Before sending large prompts, estimate token count to stay within limits:</p>
        
        <pre><code># Install tiktoken for accurate counting
npm install tiktoken</code></pre>
        
        <pre><code>import { encoding_for_model } from 'tiktoken';

function countTokens(text: string, model = 'gpt-4o-mini'): number {
    const encoder = encoding_for_model(model);
    const tokens = encoder.encode(text);
    encoder.free();  // Clean up
    return tokens.length;
}

const text = "Hello, how are you doing today?";
console.log(`Token count: ${countTokens(text)}`);  // ~7 tokens</code></pre>
    """
    content += build_section("content", "Advanced API Usage", advanced_section, "⚡")
    
    # Hands-on Exercise
    hands_on = """
        <h3>Exercise: Build a Simple Chatbot</h3>
        <p>Let's create a command-line chatbot that maintains conversation history:</p>
        
        <h4>Step 1: Create the Chat Module</h4>
        <pre><code>// src/chatbot.ts
import OpenAI from 'openai';
import * as readline from 'readline';
import 'dotenv/config';

const openai = new OpenAI();

interface Message {
    role: 'system' | 'user' | 'assistant';
    content: string;
}

// Conversation history
const messages: Message[] = [
    {
        role: 'system',
        content: `You are a helpful AI assistant. You are:
- Friendly and conversational
- Concise in your responses
    - Honest when you don't know something`
    }
];

async function chat(userInput: string): Promise&lt;string&gt; {
    // Add user message to history
    messages.push({ role: 'user', content: userInput });
    
    // Get AI response
    const response = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: messages,
        temperature: 0.7
    });
    
    const assistantMessage = response.choices[0].message.content || '';
    
    // Add assistant response to history
    messages.push({ role: 'assistant', content: assistantMessage });
    
    return assistantMessage;
}

// Main loop
async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    
    console.log('🤖 Chatbot ready! Type "quit" to exit.\\n');
    
    const askQuestion = () => {
        rl.question('You: ', async (input) => {
            if (input.toLowerCase() === 'quit') {
                console.log('Goodbye!');
                rl.close();
                return;
            }
            
            try {
                const response = await chat(input);
                console.log(`\\n🤖 Assistant: ${response}\\n`);
            } catch (error) {
                console.error('Error:', error);
            }
            
            askQuestion();  // Continue the loop
        });
    };
    
    askQuestion();
}

main();</code></pre>
        
        <h4>Step 2: Run the Chatbot</h4>
        <pre><code># Run with ts-node or after compiling
npx ts-node src/chatbot.ts</code></pre>
        
        <h4>Step 3: Test Conversation Memory</h4>
        <pre><code>🤖 Chatbot ready! Type "quit" to exit.

You: My name is Alex

🤖 Assistant: Nice to meet you, Alex! How can I help you today?

You: What's my name?

🤖 Assistant: Your name is Alex! You just told me that. 😊

You: quit
Goodbye!</code></pre>
        
        <div class="callout callout-tip">
            <strong>Why does it remember?</strong> We store all messages in the <code>messages</code> array and send the full history with each request. The model uses this context to maintain continuity.
        </div>
    """
    content += build_section("content", "Hands-On: Build a Chatbot", hands_on, "💻")
    
    # Gotchas
    gotchas = """
        <h3>Common LLM Mistakes</h3>
        
        <div class="callout callout-warning">
            <strong>🚨 Exposing API Keys:</strong> NEVER commit your OpenAI key to git! Use environment variables and add .env to .gitignore.
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 Context Window Overflow:</strong> If your conversation history grows too long, you'll hit token limits. Implement a sliding window or summarization strategy.
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 Ignoring Costs:</strong> Set spending limits in OpenAI dashboard. Log token usage in production. Use gpt-4o-mini for most tasks.
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 Trusting Output Blindly:</strong> LLMs can hallucinate (make things up). Always validate critical information, especially for code or facts.
        </div>
        
        <h3>Temperature Guide</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Temperature</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Use Case</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>0</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Factual answers, code generation, data extraction</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>0.3-0.5</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Balanced responses, Q&A, summaries</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>0.7-0.9</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Creative writing, brainstorming</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>1.0+</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Highly creative, experimental (can be incoherent)</td>
            </tr>
        </table>
    """
    content += build_section("gotchas", "Common Gotchas & Tips", gotchas, "⚠️")
    
    # Quiz
    quiz = """
        <div class="quiz-question">
            <p>1. What is a "token" in the context of LLMs?</p>
            <ul class="quiz-options">
                <li>A) A security credential</li>
                <li>B) A piece of text (roughly 4 characters)</li>
                <li>C) A complete sentence</li>
                <li>D) A paragraph</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>2. What does a temperature of 0 mean?</p>
            <ul class="quiz-options">
                <li>A) The model won't respond</li>
                <li>B) Maximum creativity</li>
                <li>C) Deterministic/consistent output</li>
                <li>D) Faster responses</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>3. Which role sets the AI's behavior in chat completions?</p>
            <ul class="quiz-options">
                <li>A) user</li>
                <li>B) assistant</li>
                <li>C) system</li>
                <li>D) admin</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>4. What prompting technique asks the model to explain its reasoning?</p>
            <ul class="quiz-options">
                <li>A) Few-shot prompting</li>
                <li>B) Chain of Thought (CoT)</li>
                <li>C) Zero-shot prompting</li>
                <li>D) Role prompting</li>
            </ul>
        </div>
        
        <p style="margin-top: 1rem; color: #8b949e;"><em>Answers: 1-B, 2-C, 3-C, 4-B</em></p>
    """
    content += build_section("quiz", "Check Your Understanding", quiz, "✅")
    
    # References
    refs = """
        <ul>
            <li>
                <a href="https://platform.openai.com/docs" target="_blank">
                    OpenAI API Documentation
                </a>
                <span style="color: #8b949e;"> - Official API reference</span>
            </li>
            <li>
                <a href="https://cookbook.openai.com/" target="_blank">
                    OpenAI Cookbook
                </a>
                <span style="color: #8b949e;"> - Example code and use cases</span>
            </li>
            <li>
                <a href="https://www.promptingguide.ai/" target="_blank">
                    Prompt Engineering Guide
                </a>
                <span style="color: #8b949e;"> - Comprehensive prompting techniques</span>
            </li>
            <li>
                <a href="https://platform.openai.com/tokenizer" target="_blank">
                    OpenAI Tokenizer
                </a>
                <span style="color: #8b949e;"> - Count tokens in your prompts</span>
            </li>
            <li>
                <a href="https://github.com/openai/openai-node" target="_blank">
                    OpenAI Node.js SDK
                </a>
                <span style="color: #8b949e;"> - Official Node.js client</span>
            </li>
        </ul>
    """
    content += build_section("references", "Sources & Further Reading", refs, "📚")
    
    # Generate HTML
    html = MODULE_TEMPLATE.format(
        title="Module 3: LLM Fundamentals",
        description="Master Large Language Models, the OpenAI API, and prompt engineering",
        time_estimate="~75 minutes",
        difficulty="Beginner",
        module_number="3",
        content=content,
        prev_link='<a href="../module-2-postgres/index.html">← Previous: Database Foundations</a>',
        next_link='<a href="../module-4-embeddings/index.html">Next: Embeddings & Vectors →</a>'
    )
    
    # Save
    output_dir = os.path.join(COURSE_DIR, "module-3-llm")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
    
    print("✅ Generated Module 3: LLM Fundamentals")


def build_module_4():
    """Build Module 4: Embeddings & Vectors"""
    
    content = ""
    
    # Objectives
    objectives = """
        <ul>
            <li>Understand what embeddings are and why they matter for AI</li>
            <li>Learn to use the OpenAI Embeddings API</li>
            <li>Master text chunking strategies for optimal retrieval</li>
            <li>Store and query embeddings using pgvector</li>
            <li>Build a complete embedding pipeline with Node.js</li>
        </ul>
    """
    content += build_section("objectives", "What You'll Learn", objectives, "🎯")
    
    # Prerequisites
    prereqs = """
        <p>Before starting this module, you should have:</p>
        <ul>
            <li>Completed Modules 1-3 (Docker, PostgreSQL, and LLM basics)</li>
            <li>pgvector extension installed (from Module 2)</li>
            <li>An OpenAI API key configured</li>
            <li>Basic understanding of vectors (we'll explain the rest!)</li>
        </ul>
        <div class="callout callout-info">
            <strong>Vector Refresher:</strong> A vector is just a list of numbers, like <code>[0.1, -0.5, 0.3, ...]</code>. Think of it as coordinates in a high-dimensional space that capture meaning.
        </div>
    """
    content += build_section("prerequisites", "Prerequisites", prereqs, "📋")
    
    # What are embeddings
    embeddings_intro = """
        <h3>What Are Embeddings?</h3>
        <p>Embeddings are numerical representations of text (or images, audio, etc.) that capture semantic meaning. They transform human-readable content into vectors that machines can compare and search.</p>
        
        <div style="background: #161b22; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
            <p style="font-family: monospace; margin: 0;">
                Text: "The cat sat on the mat"<br><br>
                ↓ Embedding Model ↓<br><br>
                Vector: [0.023, -0.156, 0.089, ..., 0.045]<br>
                (1536 dimensions for text-embedding-3-small)
            </p>
        </div>
        
        <h3>Why Embeddings Matter</h3>
        <p>Embeddings enable <strong>semantic similarity</strong> - finding content by meaning, not just keywords:</p>
        
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Query</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Keyword Match</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Semantic Match</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">"How to fix a bug"</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">❌ Docs about insects</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">✅ Debugging guides</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">"car not starting"</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">❌ Misses "vehicle won't turn on"</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">✅ Finds related content</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">"happy"</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">❌ Exact word only</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">✅ Also finds "joyful", "excited"</td>
            </tr>
        </table>
        
        <h3>How Similarity Works</h3>
        <p>Similar texts have similar vectors. We measure similarity using <strong>cosine similarity</strong> or <strong>distance</strong>:</p>
        
        <pre><code>// Conceptually:
embed("king") - embed("man") + embed("woman") ≈ embed("queen")

// Similar meanings → close vectors
similarity("cat", "kitten") → 0.92  // Very similar
similarity("cat", "dog") → 0.75     // Related (pets)
similarity("cat", "democracy") → 0.15  // Unrelated</code></pre>
        
        <div class="callout callout-tip">
            <strong>The Magic:</strong> Embeddings capture relationships! Words with similar meanings cluster together in vector space, enabling "fuzzy" semantic search.
        </div>
    """
    content += build_section("content", "Understanding Embeddings", embeddings_intro, "🧠")
    
    # OpenAI Embeddings API
    openai_embeddings = """
        <h3>OpenAI's Embedding Models</h3>
        <p>OpenAI provides several embedding models optimized for different use cases:</p>
        
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Model</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Dimensions</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Price (per 1M tokens)</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Best For</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>text-embedding-3-small</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">1536</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">$0.02</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Most use cases, best value</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>text-embedding-3-large</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">3072</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">$0.13</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Higher accuracy needs</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>text-embedding-ada-002</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">1536</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">$0.10</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Legacy (use v3 instead)</td>
            </tr>
        </table>
        
        <h3>Generating Embeddings</h3>
        <pre><code>// src/embeddings.ts
import OpenAI from 'openai';
import 'dotenv/config';

const openai = new OpenAI();

async function getEmbedding(text: string): Promise&lt;number[]&gt; {
    const response = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text
    });
    
    return response.data[0].embedding;
}

// Example usage
const embedding = await getEmbedding("What is machine learning?");
console.log(`Dimensions: ${embedding.length}`);  // 1536
console.log(`First 5 values: ${embedding.slice(0, 5)}`);</code></pre>
        
        <h3>Batch Processing</h3>
        <p>For efficiency, embed multiple texts in a single API call:</p>
        
        <pre><code>async function getEmbeddings(texts: string[]): Promise&lt;number[][]&gt; {
    const response = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: texts  // Array of strings
    });
    
    // Sort by index to maintain order
    return response.data
        .sort((a, b) => a.index - b.index)
        .map(item => item.embedding);
}

// Embed multiple texts at once
const texts = [
    "Machine learning is a subset of AI",
    "Neural networks mimic the human brain",
    "Deep learning uses multiple layers"
];

const embeddings = await getEmbeddings(texts);
console.log(`Generated ${embeddings.length} embeddings`);</code></pre>
        
        <div class="callout callout-info">
            <strong>Batch Limits:</strong> You can send up to 2048 texts per request, with a combined limit of ~8191 tokens. For large datasets, batch in groups of 100-500 texts.
        </div>
    """
    content += build_section("content", "OpenAI Embeddings API", openai_embeddings, "🔌")
    
    # Chunking strategies
    chunking_section = """
        <h3>Why Chunking Matters</h3>
        <p>Large documents must be split into smaller chunks before embedding. Here's why:</p>
        
        <ul>
            <li><strong>Token Limits:</strong> Embedding models have input limits (~8191 tokens)</li>
            <li><strong>Precision:</strong> Smaller chunks = more precise retrieval</li>
            <li><strong>Context Windows:</strong> Retrieved chunks must fit in LLM context</li>
            <li><strong>Relevance:</strong> A 10,000-word doc about many topics won't match queries well</li>
        </ul>
        
        <h3>Chunking Strategies</h3>
        
        <h4>1. Fixed-Size Chunking</h4>
        <p>Split by character/token count with overlap:</p>
        
        <pre><code>function chunkBySize(text: string, chunkSize: number, overlap: number): string[] {
    const chunks: string[] = [];
    let start = 0;
    
    while (start < text.length) {
        const end = start + chunkSize;
        chunks.push(text.slice(start, end));
        start = end - overlap;  // Step back by overlap amount
    }
    
    return chunks;
}

// Example: 500 char chunks with 50 char overlap
const chunks = chunkBySize(longDocument, 500, 50);</code></pre>
        
        <h4>2. Sentence-Based Chunking</h4>
        <p>Split on sentence boundaries, then combine until size limit:</p>
        
        <pre><code>function chunkBySentences(text: string, maxChunkSize: number): string[] {
    // Split into sentences (simplified regex)
    const sentences = text.split(/(?&lt;=[.!?])\\s+/);
    
    const chunks: string[] = [];
    let currentChunk = '';
    
    for (const sentence of sentences) {
        if ((currentChunk + sentence).length > maxChunkSize && currentChunk) {
            chunks.push(currentChunk.trim());
            currentChunk = sentence;
        } else {
            currentChunk += ' ' + sentence;
        }
    }
    
    if (currentChunk.trim()) {
        chunks.push(currentChunk.trim());
    }
    
    return chunks;
}</code></pre>
        
        <h4>3. Semantic Chunking (Advanced)</h4>
        <p>Split by document structure - headers, paragraphs, sections:</p>
        
        <pre><code>function chunkByStructure(markdown: string): string[] {
    // Split on headers while keeping them
    const sections = markdown.split(/(?=^#{1,3}\\s)/m);
    
    return sections
        .map(s => s.trim())
        .filter(s => s.length > 0);
}</code></pre>
        
        <h3>Overlap: The Secret Sauce</h3>
        <p>Overlap ensures context isn't lost at chunk boundaries:</p>
        
        <div style="background: #161b22; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
            <p style="font-family: monospace; margin: 0; font-size: 0.85rem;">
                Without overlap:<br>
                Chunk 1: "The capital of France is"|<br>
                Chunk 2: |"Paris. It's known for..."<br>
                ❌ Query "France capital" might miss "Paris"<br><br>
                
                With 20% overlap:<br>
                Chunk 1: "The capital of France is Paris."<br>
                Chunk 2: "France is Paris. It's known for..."<br>
                ✅ Query "France capital" finds "Paris" in both
            </p>
        </div>
        
        <h3>Recommended Settings</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Content Type</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Chunk Size</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Overlap</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Technical docs</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">500-1000 tokens</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">10-20%</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Q&A / FAQ</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">200-500 tokens</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">0-10%</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Long-form content</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">1000-2000 tokens</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">15-25%</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Code</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Function/class level</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Include imports</td>
            </tr>
        </table>
        
        <div class="callout callout-tip">
            <strong>Pro Tip:</strong> Start with 500 tokens and 10% overlap. Adjust based on retrieval quality - if answers are incomplete, try larger chunks; if irrelevant results appear, try smaller chunks.
        </div>
    """
    content += build_section("content", "Text Chunking Strategies", chunking_section, "✂️")
    
    # Storing in pgvector
    pgvector_section = """
        <h3>Setting Up pgvector Storage</h3>
        <p>Let's create a table to store our embeddings with pgvector:</p>
        
        <pre><code>-- Create the documents table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),  -- Match your model's dimensions
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create an index for fast similarity search
CREATE INDEX ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- Adjust based on data size</code></pre>
        
        <h3>Storing Embeddings</h3>
        <pre><code>// src/store.ts
import { Pool } from 'pg';
import OpenAI from 'openai';

const pool = new Pool({
    connectionString: process.env.DATABASE_URL
});

const openai = new OpenAI();

async function storeDocument(content: string, metadata: object = {}) {
    // Generate embedding
    const response = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: content
    });
    const embedding = response.data[0].embedding;
    
    // Store in database
    const result = await pool.query(
        `INSERT INTO documents (content, embedding, metadata)
         VALUES ($1, $2, $3)
         RETURNING id`,
        [content, JSON.stringify(embedding), metadata]
    );
    
    return result.rows[0].id;
}

// Store a document
const id = await storeDocument(
    "PostgreSQL is a powerful open-source database",
    { source: "docs", category: "databases" }
);
console.log(`Stored document with ID: ${id}`);</code></pre>
        
        <h3>Semantic Search</h3>
        <pre><code>async function searchSimilar(query: string, limit = 5) {
    // Embed the query
    const response = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: query
    });
    const queryEmbedding = response.data[0].embedding;
    
    // Find similar documents using cosine distance
    const result = await pool.query(
        `SELECT id, content, metadata,
                1 - (embedding &lt;=&gt; $1) as similarity
         FROM documents
         ORDER BY embedding &lt;=&gt; $1
         LIMIT $2`,
        [JSON.stringify(queryEmbedding), limit]
    );
    
    return result.rows;
}

// Search for similar content
const results = await searchSimilar("relational database systems");
for (const doc of results) {
    console.log(`[${doc.similarity.toFixed(3)}] ${doc.content}`);
}</code></pre>
        
        <h3>Distance Operators</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Operator</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Name</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Use Case</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>&lt;=&gt;</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Cosine distance</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Text similarity (recommended)</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>&lt;-&gt;</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">L2 (Euclidean)</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Image embeddings</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><code>&lt;#&gt;</code></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Inner product</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Normalized vectors</td>
            </tr>
        </table>
        
        <div class="callout callout-info">
            <strong>Cosine vs L2:</strong> Cosine distance measures angle (direction), L2 measures magnitude. For text, use cosine - it's invariant to document length.
        </div>
    """
    content += build_section("content", "Storing Embeddings in pgvector", pgvector_section, "🗄️")
    
    # Hands-on Exercise
    hands_on = """
        <h3>Exercise: Build an Embedding Pipeline</h3>
        <p>Let's create a complete pipeline that chunks documents, generates embeddings, and enables semantic search:</p>
        
        <h4>Step 1: Set Up the Database</h4>
        <pre><code>-- Run in psql or database client
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    source VARCHAR(255),
    chunk_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);</code></pre>
        
        <h4>Step 2: Create the Embedding Service</h4>
        <pre><code>// src/embedding-service.ts
import OpenAI from 'openai';
import { Pool } from 'pg';
import 'dotenv/config';

const openai = new OpenAI();
const pool = new Pool({ connectionString: process.env.DATABASE_URL });

// Chunking function
function chunkText(text: string, maxLength = 1000, overlap = 100): string[] {
    const chunks: string[] = [];
    let start = 0;
    
    while (start < text.length) {
        let end = start + maxLength;
        
        // Try to break at sentence boundary
        if (end < text.length) {
            const lastPeriod = text.lastIndexOf('.', end);
            if (lastPeriod > start + maxLength * 0.5) {
                end = lastPeriod + 1;
            }
        }
        
        chunks.push(text.slice(start, end).trim());
        start = end - overlap;
    }
    
    return chunks.filter(c => c.length > 0);
}

// Generate embedding
async function embed(text: string): Promise&lt;number[]&gt; {
    const response = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text
    });
    return response.data[0].embedding;
}

// Process and store a document
export async function ingestDocument(content: string, source: string) {
    const chunks = chunkText(content);
    console.log(`Processing ${chunks.length} chunks from ${source}`);
    
    for (let i = 0; i < chunks.length; i++) {
        const embedding = await embed(chunks[i]);
        
        await pool.query(
            `INSERT INTO documents (content, embedding, source, chunk_index)
             VALUES ($1, $2, $3, $4)`,
            [chunks[i], JSON.stringify(embedding), source, i]
        );
        
        console.log(`  Stored chunk ${i + 1}/${chunks.length}`);
    }
    
    return chunks.length;
}

// Semantic search
export async function search(query: string, limit = 5) {
    const queryEmbedding = await embed(query);
    
    const result = await pool.query(
        `SELECT content, source, chunk_index,
                1 - (embedding &lt;=&gt; $1) as similarity
         FROM documents
         ORDER BY embedding &lt;=&gt; $1
         LIMIT $2`,
        [JSON.stringify(queryEmbedding), limit]
    );
    
    return result.rows;
}</code></pre>
        
        <h4>Step 3: Test the Pipeline</h4>
        <pre><code>// src/test-embeddings.ts
import { ingestDocument, search } from './embedding-service';

async function main() {
    // Sample documents to ingest
    const docs = [
        {
            content: `PostgreSQL is a powerful, open source object-relational 
            database system with over 35 years of active development. It has 
            earned a strong reputation for reliability, feature robustness, 
            and performance. PostgreSQL supports both SQL and JSON querying.`,
            source: 'postgres-intro'
        },
        {
            content: `Docker is a platform for developing, shipping, and running 
            applications in containers. Containers are lightweight, standalone 
            packages that include everything needed to run an application: code, 
            runtime, system tools, and libraries.`,
            source: 'docker-intro'
        },
        {
            content: `Machine learning is a subset of artificial intelligence 
            that enables systems to learn and improve from experience without 
            being explicitly programmed. It focuses on developing algorithms 
            that can access data and use it to learn for themselves.`,
            source: 'ml-intro'
        }
    ];
    
    // Ingest documents
    console.log('📥 Ingesting documents...\\n');
    for (const doc of docs) {
        await ingestDocument(doc.content, doc.source);
    }
    
    // Test searches
    console.log('\\n🔍 Testing semantic search...\\n');
    
    const queries = [
        "relational database with JSON support",
        "containerization technology",
        "AI systems that learn from data"
    ];
    
    for (const query of queries) {
        console.log(`Query: "${query}"`);
        const results = await search(query, 2);
        
        for (const r of results) {
            console.log(`  [${r.similarity.toFixed(3)}] ${r.source}: ${r.content.slice(0, 60)}...`);
        }
        console.log();
    }
}

main().catch(console.error);</code></pre>
        
        <h4>Step 4: Run and Verify</h4>
        <pre><code># Run the test
npx ts-node src/test-embeddings.ts

# Expected output:
📥 Ingesting documents...

Processing 1 chunks from postgres-intro
  Stored chunk 1/1
Processing 1 chunks from docker-intro
  Stored chunk 1/1
Processing 1 chunks from ml-intro
  Stored chunk 1/1

🔍 Testing semantic search...

Query: "relational database with JSON support"
  [0.847] postgres-intro: PostgreSQL is a powerful, open source object-relational...

Query: "containerization technology"
  [0.812] docker-intro: Docker is a platform for developing, shipping, and running...</code></pre>
        
        <div class="callout callout-tip">
            <strong>Success!</strong> Notice how semantic search finds relevant content even when query words don't exactly match the document text!
        </div>
    """
    content += build_section("content", "Hands-On: Build an Embedding Pipeline", hands_on, "💻")
    
    # Gotchas
    gotchas = """
        <h3>Common Embedding Mistakes</h3>
        
        <div class="callout callout-warning">
            <strong>🚨 Mixing Embedding Models:</strong> Never mix embeddings from different models! A vector from <code>text-embedding-3-small</code> is incompatible with <code>ada-002</code>. If you switch models, re-embed everything.
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 Chunks Too Large:</strong> Giant chunks dilute semantic meaning. A 5000-word chunk about 20 topics won't match specific queries well. Aim for focused, coherent chunks.
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 Chunks Too Small:</strong> Tiny chunks lose context. "It is blue" means nothing without knowing what "it" refers to. Include enough context for standalone meaning.
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 No Overlap:</strong> Without overlap, you'll miss content at chunk boundaries. Always use 10-20% overlap for continuous text.
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 Forgetting Metadata:</strong> Store source, page numbers, timestamps with embeddings. You'll need this for citations, filtering, and debugging.
        </div>
        
        <h3>Performance Tips</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Issue</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Solution</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Slow embedding API calls</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Batch texts (up to 2048 per request)</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Slow similarity search</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Add IVFFlat or HNSW index</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">High storage costs</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Use smaller model or reduce dimensions</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Poor search quality</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Experiment with chunk sizes</td>
            </tr>
        </table>
    """
    content += build_section("gotchas", "Common Gotchas & Tips", gotchas, "⚠️")
    
    # Quiz
    quiz = """
        <div class="quiz-question">
            <p>1. What are embeddings?</p>
            <ul class="quiz-options">
                <li>A) Compressed text files</li>
                <li>B) Numerical vectors that capture semantic meaning</li>
                <li>C) Database indexes</li>
                <li>D) Encryption keys</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>2. How many dimensions does text-embedding-3-small produce?</p>
            <ul class="quiz-options">
                <li>A) 256</li>
                <li>B) 768</li>
                <li>C) 1536</li>
                <li>D) 3072</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>3. Why do we use overlap when chunking?</p>
            <ul class="quiz-options">
                <li>A) To save storage space</li>
                <li>B) To avoid losing context at chunk boundaries</li>
                <li>C) To speed up embedding generation</li>
                <li>D) To reduce API costs</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>4. Which pgvector operator is recommended for text similarity?</p>
            <ul class="quiz-options">
                <li>A) &lt;-&gt; (L2 distance)</li>
                <li>B) &lt;=&gt; (cosine distance)</li>
                <li>C) &lt;#&gt; (inner product)</li>
                <li>D) = (equality)</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>5. What happens if you mix embeddings from different models?</p>
            <ul class="quiz-options">
                <li>A) They automatically convert</li>
                <li>B) Search results will be meaningless/incorrect</li>
                <li>C) Performance improves</li>
                <li>D) Nothing, they're compatible</li>
            </ul>
        </div>
        
        <p style="margin-top: 1rem; color: #8b949e;"><em>Answers: 1-B, 2-C, 3-B, 4-B, 5-B</em></p>
    """
    content += build_section("quiz", "Check Your Understanding", quiz, "✅")
    
    # References
    refs = """
        <ul>
            <li>
                <a href="https://platform.openai.com/docs/guides/embeddings" target="_blank">
                    OpenAI Embeddings Guide
                </a>
                <span style="color: #8b949e;"> - Official documentation</span>
            </li>
            <li>
                <a href="https://github.com/pgvector/pgvector" target="_blank">
                    pgvector GitHub
                </a>
                <span style="color: #8b949e;"> - Vector extension for PostgreSQL</span>
            </li>
            <li>
                <a href="https://www.pinecone.io/learn/chunking-strategies/" target="_blank">
                    Chunking Strategies for LLM Applications
                </a>
                <span style="color: #8b949e;"> - Deep dive into chunking</span>
            </li>
            <li>
                <a href="https://cookbook.openai.com/examples/embedding_long_inputs" target="_blank">
                    OpenAI Cookbook: Embedding Long Inputs
                </a>
                <span style="color: #8b949e;"> - Handling large documents</span>
            </li>
            <li>
                <a href="https://supabase.com/docs/guides/ai/vector-columns" target="_blank">
                    Supabase Vector Guide
                </a>
                <span style="color: #8b949e;"> - pgvector with Supabase</span>
            </li>
        </ul>
    """
    content += build_section("references", "Sources & Further Reading", refs, "📚")
    
    # Generate HTML
    html = MODULE_TEMPLATE.format(
        title="Module 4: Embeddings & Vectors",
        description="Transform text into vectors for semantic search",
        time_estimate="~60 minutes",
        difficulty="Intermediate",
        module_number="4",
        content=content,
        prev_link='<a href="../module-3-llm/index.html">← Previous: LLM Fundamentals</a>',
        next_link='<a href="../module-5-rag/index.html">Next: RAG Pipeline →</a>'
    )
    
    # Save
    output_dir = os.path.join(COURSE_DIR, "module-4-embeddings")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
    
    print("✅ Generated Module 4: Embeddings & Vectors")


def build_module_5():
    """Build Module 5: RAG Pipeline - The Complete System"""
    
    content = ""
    
    # Objectives
    objectives = """
        <ul>
            <li>Understand how RAG combines retrieval with generation</li>
            <li>Build a complete RAG pipeline from scratch</li>
            <li>Implement document ingestion, retrieval, and response generation</li>
            <li>Create API endpoints for your RAG chatbot</li>
            <li>Handle conversation context and follow-up questions</li>
        </ul>
    """
    content += build_section("objectives", "What You'll Learn", objectives, "🎯")
    
    # Prerequisites
    prereqs = """
        <p>Before starting this module, you should have completed:</p>
        <ul>
            <li>Module 2: Database with pgvector set up</li>
            <li>Module 3: OpenAI API working</li>
            <li>Module 4: Embeddings and semantic search implemented</li>
        </ul>
        <div class="callout callout-info">
            <strong>This is it!</strong> This module ties everything together into a working RAG system.
        </div>
        <div class="callout callout-tip">
            <strong>If SQL is new:</strong> You don’t need “deep SQL” here. Make sure you completed Module 2’s <em>SQL Cheat Sheet</em> and <em>Mini-Lab</em>, then come back.
        </div>
    """
    content += build_section("prerequisites", "Prerequisites", prereqs, "📋")
    
    # RAG Architecture
    rag_intro = """
        <h3>What is RAG?</h3>
        <p><strong>Retrieval-Augmented Generation (RAG)</strong> is a technique that enhances LLM responses by providing relevant context from your own data. Instead of relying solely on what the model was trained on, you give it specific information to answer from.</p>
        
        <div style="background: #161b22; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; text-align: center;">
            <p style="font-family: monospace; font-size: 1.1rem; margin: 0; line-height: 2;">
                User Question<br>
                ↓<br>
                <span style="color: #58a6ff;">Embed Question → Search Database → Get Relevant Chunks</span><br>
                ↓<br>
                <span style="color: #a371f7;">Build Prompt with Context + Question</span><br>
                ↓<br>
                <span style="color: #3fb950;">Send to LLM → Generate Answer</span><br>
                ↓<br>
                Response to User
            </p>
        </div>
        
        <h3>Why RAG Beats Fine-Tuning</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Approach</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Pros</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Cons</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><strong>RAG</strong></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d; color: #3fb950;">
                    • Easy to update data<br>
                    • No training needed<br>
                    • Traceable sources<br>
                    • Works immediately
                </td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d; color: #f87171;">
                    • Retrieval can miss<br>
                    • Needs good chunking
                </td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><strong>Fine-Tuning</strong></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d; color: #3fb950;">
                    • Learns patterns<br>
                    • No runtime retrieval
                </td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d; color: #f87171;">
                    • Expensive to train<br>
                    • Hard to update<br>
                    • Can hallucinate<br>
                    • No source citations
                </td>
            </tr>
        </table>
        
        <div class="callout callout-tip">
            <strong>Rule of thumb:</strong> Use RAG for factual Q&A from your docs. Use fine-tuning for style/tone changes or specialized reasoning patterns.
        </div>
    """
    content += build_section("content", "RAG Architecture Overview", rag_intro, "🏗️")
    
    # Complete RAG Implementation
    rag_impl = """
        <h3>The Complete RAG Service</h3>
        <p>Let's build a complete RAG service that handles everything:</p>
        
        <pre><code>// src/rag.ts
import OpenAI from 'openai';
import pool from './db.js';

const openai = new OpenAI();

interface RAGResponse {
    answer: string;
    sources: Array<{
        title: string;
        content: string;
        similarity: number;
    }>;
    tokensUsed: number;
}

// The main RAG function
export async function askRAG(
    question: string,
    options: {
        topK?: number;
        threshold?: number;
        systemPrompt?: string;
    } = {}
): Promise&lt;RAGResponse&gt; {
    const { topK = 5, threshold = 0.7, systemPrompt } = options;
    
    // Step 1: Embed the question
    console.log('🔍 Embedding question...');
    const embeddingResponse = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: question
    });
    const questionEmbedding = embeddingResponse.data[0].embedding;
    
    // Step 2: Search for relevant chunks
    console.log('📚 Searching for relevant documents...');
    const searchResult = await pool.query(\\`
        SELECT 
            c.content,
            d.title,
            1 - (c.embedding &lt;=&gt; $1::vector) as similarity
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE 1 - (c.embedding &lt;=&gt; $1::vector) > $2
        ORDER BY c.embedding &lt;=&gt; $1::vector
        LIMIT $3
    \\`, [JSON.stringify(questionEmbedding), threshold, topK]);
    
    const sources = searchResult.rows;
    console.log(\\`  Found \\${sources.length} relevant chunks\\`);
    
    // Step 3: Build the context
    const context = sources.length > 0
        ? sources.map((s, i) => 
            \\`[Source \\${i + 1}: \\${s.title}]\\n\\${s.content}\\`
          ).join('\\n\\n---\\n\\n')
        : 'No relevant documents found.';
    
    // Step 4: Build the prompt
    const defaultSystemPrompt = \\`You are a helpful assistant that answers questions based on the provided context.

Rules:
- Only answer based on the provided context
- If the context doesn't contain the answer, say "I don't have information about that in my knowledge base"
- Be concise but thorough
- Cite sources when possible using [Source N] format\\`;
    
    const messages: OpenAI.ChatCompletionMessageParam[] = [
        { 
            role: 'system', 
            content: systemPrompt || defaultSystemPrompt 
        },
        { 
            role: 'user', 
            content: \\`Context:
\\${context}

Question: \\${question}

Please answer based on the context above.\\`
        }
    ];
    
    // Step 5: Generate the response
    console.log('🤖 Generating response...');
    const completion = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages,
        temperature: 0.3,  // Lower for more factual responses
        max_tokens: 1000
    });
    
    return {
        answer: completion.choices[0].message.content || 'No response generated',
        sources: sources.map(s => ({
            title: s.title,
            content: s.content.slice(0, 200) + '...',
            similarity: parseFloat(s.similarity)
        })),
        tokensUsed: completion.usage?.total_tokens || 0
    };
}</code></pre>
        
        <h3>Test It Out</h3>
        <pre><code>// scripts/test-rag.ts
import { askRAG } from '../src/rag.js';
import pool from '../src/db.js';

async function main() {
    const questions = [
        'How do I reset my password?',
        'What are the API rate limits?',
        'How do I authenticate with the API?'
    ];
    
    for (const q of questions) {
        console.log('\\n' + '='.repeat(60));
        console.log('❓ Question:', q);
        console.log('='.repeat(60));
        
        const result = await askRAG(q);
        
        console.log('\\n📝 Answer:');
        console.log(result.answer);
        
        console.log('\\n📚 Sources used:');
        result.sources.forEach((s, i) => {
            console.log(\\`  [\\${i + 1}] \\${s.title} (\\${(s.similarity * 100).toFixed(0)}% match)\\`);
        });
        
        console.log(\\`\\n📊 Tokens used: \\${result.tokensUsed}\\`);
    }
    
    await pool.end();
}

main();</code></pre>
    """
    content += build_section("content", "Building the RAG Pipeline", rag_impl, "⚡")
    
    # API Endpoints
    api_section = """
        <h3>Creating REST API Endpoints</h3>
        <p>Let's expose our RAG system through a clean API:</p>
        
        <pre><code>// src/routes/chat.ts
import { Router, Request, Response } from 'express';
import { askRAG } from '../rag.js';

const router = Router();

interface ChatRequest {
    message: string;
    conversationId?: string;
}

// Simple Q&A endpoint
router.post('/ask', async (req: Request, res: Response) => {
    try {
        const { message } = req.body as ChatRequest;
        
        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }
        
        const result = await askRAG(message);
        
        res.json({
            success: true,
            answer: result.answer,
            sources: result.sources,
            tokensUsed: result.tokensUsed
        });
    } catch (error) {
        console.error('RAG error:', error);
        res.status(500).json({ error: 'Failed to process question' });
    }
});

export default router;</code></pre>
        
        <h3>Wire It Up in Express</h3>
        <pre><code>// src/index.ts
import express from 'express';
import cors from 'cors';
import chatRoutes from './routes/chat.js';
import 'dotenv/config';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Routes
app.use('/api/chat', chatRoutes);

// Health check
app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
    console.log(\\`🚀 RAG API running on http://localhost:\\${PORT}\\`);
});

export default app;</code></pre>
        
        <h3>Test with cURL</h3>
        <pre><code># Ask a question
curl -X POST http://localhost:3000/api/chat/ask \\
  -H "Content-Type: application/json" \\
  -d '{"message": "How do I reset my password?"}'

# Response:
{
  "success": true,
  "answer": "To reset your password, click the 'Forgot Password' link on the login page. You'll receive an email with a reset link that expires after 24 hours. [Source 1]",
  "sources": [
    {
      "title": "Password Reset",
      "content": "If you forgot your password, click the Forgot Password link...",
      "similarity": 0.91
    }
  ],
  "tokensUsed": 245
}</code></pre>
    """
    content += build_section("content", "API Endpoints", api_section, "🔌")
    
    # Conversation Memory
    memory_section = """
        <h3>Adding Conversation Memory</h3>
        <p>For a true chatbot experience, we need to remember previous messages:</p>
        
        <pre><code>// src/conversation.ts
import OpenAI from 'openai';

interface Message {
    role: 'user' | 'assistant';
    content: string;
}

// In-memory store (use Redis for production)
const conversations = new Map&lt;string, Message[]&gt;();

export function getConversation(id: string): Message[] {
    return conversations.get(id) || [];
}

export function addMessage(id: string, message: Message): void {
    const history = getConversation(id);
    history.push(message);
    
    // Keep last 10 messages to manage context size
    if (history.length > 10) {
        history.splice(0, history.length - 10);
    }
    
    conversations.set(id, history);
}

export function clearConversation(id: string): void {
    conversations.delete(id);
}</code></pre>
        
        <h3>RAG with Conversation Context</h3>
        <pre><code>// Enhanced askRAG with conversation history
export async function askRAGWithHistory(
    question: string,
    conversationId: string,
    options: { topK?: number; threshold?: number } = {}
): Promise&lt;RAGResponse&gt; {
    const { topK = 5, threshold = 0.7 } = options;
    
    // Get conversation history
    const history = getConversation(conversationId);
    
    // Embed and search (same as before)
    const embeddingResponse = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: question
    });
    const questionEmbedding = embeddingResponse.data[0].embedding;
    
    const searchResult = await pool.query(\\`
        SELECT c.content, d.title,
               1 - (c.embedding &lt;=&gt; $1::vector) as similarity
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE 1 - (c.embedding &lt;=&gt; $1::vector) > $2
        ORDER BY c.embedding &lt;=&gt; $1::vector
        LIMIT $3
    \\`, [JSON.stringify(questionEmbedding), threshold, topK]);
    
    const sources = searchResult.rows;
    const context = sources.map((s, i) => 
        \\`[Source \\${i + 1}]\\n\\${s.content}\\`
    ).join('\\n\\n');
    
    // Build messages with history
    const messages: OpenAI.ChatCompletionMessageParam[] = [
        { 
            role: 'system', 
            content: \\`You are a helpful assistant. Answer based on the context provided.
            
Context from knowledge base:
\\${context || 'No relevant documents found.'}\\`
        },
        // Include conversation history
        ...history.map(m => ({
            role: m.role as 'user' | 'assistant',
            content: m.content
        })),
        // Current question
        { role: 'user', content: question }
    ];
    
    const completion = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages,
        temperature: 0.3,
        max_tokens: 1000
    });
    
    const answer = completion.choices[0].message.content || '';
    
    // Save to conversation history
    addMessage(conversationId, { role: 'user', content: question });
    addMessage(conversationId, { role: 'assistant', content: answer });
    
    return {
        answer,
        sources: sources.map(s => ({
            title: s.title,
            content: s.content.slice(0, 200) + '...',
            similarity: parseFloat(s.similarity)
        })),
        tokensUsed: completion.usage?.total_tokens || 0
    };
}</code></pre>
        
        <div class="callout callout-tip">
            <strong>Follow-up questions work!</strong> Now users can ask "What about the rate limits?" and the bot remembers they were asking about the API.
        </div>
    """
    content += build_section("content", "Conversation Memory", memory_section, "💭")
    
    # Hands-on
    hands_on = """
        <h3>Complete Exercise: Build Your RAG Chatbot</h3>
        
        <h4>Step 1: Project Structure</h4>
        <pre><code>my-rag-app/
├── src/
│   ├── index.ts         # Express server
│   ├── db.ts            # Database connection
│   ├── rag.ts           # RAG logic
│   ├── conversation.ts  # Memory management
│   └── routes/
│       └── chat.ts      # API routes
├── scripts/
│   ├── ingest.ts        # Document ingestion
│   └── test-rag.ts      # Testing
├── package.json
├── tsconfig.json
└── .env</code></pre>
        
        <h4>Step 2: Ingest Your Documents</h4>
        <pre><code>// Run ingestion
npx tsx scripts/ingest.ts

// Output:
// Processing: Getting Started
//   Created 3 chunks
//   Generated embeddings
//   Stored in database
// ✅ Ingestion complete!</code></pre>
        
        <h4>Step 3: Start the Server</h4>
        <pre><code>npx tsx src/index.ts

// 🚀 RAG API running on http://localhost:3000</code></pre>
        
        <h4>Step 4: Test Your Chatbot</h4>
        <pre><code># First question
curl -X POST http://localhost:3000/api/chat/ask \\
  -H "Content-Type: application/json" \\
  -d '{"message": "How do I get started?", "conversationId": "user-123"}'

# Follow-up question (remembers context)
curl -X POST http://localhost:3000/api/chat/ask \\
  -H "Content-Type: application/json" \\
  -d '{"message": "What about authentication?", "conversationId": "user-123"}'</code></pre>
        
        <div class="callout callout-tip">
            <strong>🎉 Congratulations!</strong> You've built a complete RAG chatbot from scratch! It can:
            <ul style="margin-top: 0.5rem;">
                <li>Ingest and chunk documents</li>
                <li>Generate embeddings and store them</li>
                <li>Search semantically for relevant content</li>
                <li>Generate grounded responses with citations</li>
                <li>Remember conversation context</li>
            </ul>
        </div>
    """
    content += build_section("content", "Hands-On: Complete RAG Chatbot", hands_on, "💻")
    
    # Gotchas
    gotchas = """
        <h3>Common RAG Mistakes</h3>
        
        <div class="callout callout-warning">
            <strong>🚨 Not Setting Temperature Low:</strong> For factual Q&A, use temperature 0.1-0.3. Higher values cause creative "interpretation" of facts.
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 Ignoring Empty Results:</strong> Always handle the case when no relevant documents are found. Don't let the LLM make things up!
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 Too Much Context:</strong> Stuffing too many chunks into the prompt dilutes relevance. 3-5 highly relevant chunks > 20 mediocre ones.
        </div>
        
        <div class="callout callout-warning">
            <strong>🚨 No Source Attribution:</strong> Always include sources so users can verify. This builds trust and catches hallucinations.
        </div>
        
        <h3>Production Tips</h3>
        <ul>
            <li><strong>Cache embeddings:</strong> Don't re-embed the same question repeatedly</li>
            <li><strong>Use streaming:</strong> For better UX, stream responses to the client</li>
            <li><strong>Monitor costs:</strong> Track token usage per conversation</li>
            <li><strong>Implement fallbacks:</strong> Have graceful degradation when API fails</li>
            <li><strong>Log everything:</strong> Questions, retrieved chunks, and responses for debugging</li>
        </ul>
    """
    content += build_section("gotchas", "Common Gotchas & Tips", gotchas, "⚠️")
    
    # Quiz
    quiz = """
        <div class="quiz-question">
            <p>1. What does RAG stand for?</p>
            <ul class="quiz-options">
                <li>A) Random Augmented Generation</li>
                <li>B) Retrieval-Augmented Generation</li>
                <li>C) Rapid AI Generation</li>
                <li>D) Recursive Answer Generation</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>2. In a RAG pipeline, what happens FIRST when a user asks a question?</p>
            <ul class="quiz-options">
                <li>A) Send the question directly to the LLM</li>
                <li>B) Embed the question and search for relevant documents</li>
                <li>C) Generate a response immediately</li>
                <li>D) Store the question in the database</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>3. Why is low temperature (0.1-0.3) recommended for RAG?</p>
            <ul class="quiz-options">
                <li>A) To make responses faster</li>
                <li>B) To reduce API costs</li>
                <li>C) To keep responses factual and consistent</li>
                <li>D) To make responses more creative</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>4. What's the main advantage of RAG over fine-tuning?</p>
            <ul class="quiz-options">
                <li>A) Faster responses</li>
                <li>B) Easy to update data without retraining</li>
                <li>C) Lower memory usage</li>
                <li>D) Works offline</li>
            </ul>
        </div>
        
        <p style="margin-top: 1rem; color: #8b949e;"><em>Answers: 1-B, 2-B, 3-C, 4-B</em></p>
    """
    content += build_section("quiz", "Check Your Understanding", quiz, "✅")
    
    # References
    refs = """
        <ul>
            <li>
                <a href="https://python.langchain.com/docs/tutorials/rag/" target="_blank">
                    LangChain RAG Tutorial
                </a>
                <span style="color: #8b949e;"> - Official LangChain guide</span>
            </li>
            <li>
                <a href="https://www.pinecone.io/learn/chunking-strategies/" target="_blank">
                    Chunking Strategies for LLM Applications
                </a>
                <span style="color: #8b949e;"> - Pinecone's comprehensive guide</span>
            </li>
            <li>
                <a href="https://www.anthropic.com/news/contextual-retrieval" target="_blank">
                    Anthropic: Contextual Retrieval
                </a>
                <span style="color: #8b949e;"> - Advanced RAG techniques</span>
            </li>
            <li>
                <a href="https://platform.openai.com/docs/guides/text-generation" target="_blank">
                    OpenAI Chat Completions Guide
                </a>
                <span style="color: #8b949e;"> - Official API documentation</span>
            </li>
            <li>
                <a href="https://github.com/pgvector/pgvector" target="_blank">
                    pgvector Documentation
                </a>
                <span style="color: #8b949e;"> - Vector storage in PostgreSQL</span>
            </li>
        </ul>
    """
    content += build_section("references", "Sources & Further Reading", refs, "📚")
    
    # Generate HTML
    html = MODULE_TEMPLATE.format(
        title="Module 5: RAG Pipeline",
        description="Build a complete Retrieval-Augmented Generation system",
        time_estimate="~120 minutes",
        difficulty="Advanced",
        module_number="5",
        content=content,
        prev_link='<a href="../module-4-embeddings/index.html">← Previous: Embeddings & Vectors</a>',
        next_link='<a href="../module-6-production/index.html">Next: Production & Polish →</a>'
    )
    
    # Save
    output_dir = os.path.join(COURSE_DIR, "module-5-rag")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
    
    print("✅ Generated Module 5: RAG Pipeline")


def build_module_6():
    """Build Module 6: Production & Polish"""
    
    content = ""
    
    # Objectives
    objectives = """
        <ul>
            <li>Add streaming responses for better user experience</li>
            <li>Implement error handling and fallbacks</li>
            <li>Set up logging and monitoring</li>
            <li>Optimize performance and costs</li>
            <li>Deploy your RAG application</li>
        </ul>
    """
    content += build_section("objectives", "What You'll Learn", objectives, "🎯")
    
    # Prerequisites
    prereqs = """
        <p>Before starting this module, you should have:</p>
        <ul>
            <li>Completed Module 5: Working RAG pipeline</li>
            <li>Basic understanding of deployment concepts</li>
        </ul>
        <div class="callout callout-info">
            <strong>Final Module!</strong> This takes your RAG app from "works on my machine" to production-ready.
        </div>
    """
    content += build_section("prerequisites", "Prerequisites", prereqs, "📋")
    
    # Streaming
    streaming_section = """
        <h3>Why Streaming Matters</h3>
        <p>Without streaming, users stare at a blank screen for 5-10 seconds. With streaming, they see the response build in real-time - much better UX!</p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
            <div style="background: #21262d; padding: 1rem; border-radius: 8px; border: 1px solid #f87171;">
                <strong style="color: #f87171;">❌ Without Streaming</strong>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">User clicks → Waits 8 seconds → Full response appears</p>
            </div>
            <div style="background: #21262d; padding: 1rem; border-radius: 8px; border: 1px solid #3fb950;">
                <strong style="color: #3fb950;">✅ With Streaming</strong>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">User clicks → Words appear immediately → Builds naturally</p>
            </div>
        </div>
        
        <h3>Implementing Streaming</h3>
        <pre><code>// src/rag-stream.ts
import OpenAI from 'openai';

export async function* askRAGStream(
    question: string,
    context: string
): AsyncGenerator&lt;string&gt; {
    const openai = new OpenAI();
    
    const stream = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
            { 
                role: 'system', 
                content: \\`Answer based on the context:\\n\\n\\${context}\\` 
            },
            { role: 'user', content: question }
        ],
        stream: true  // Enable streaming!
    });
    
    for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content;
        if (content) {
            yield content;
        }
    }
}</code></pre>
        
        <h3>Streaming API Endpoint</h3>
        <pre><code>// src/routes/chat.ts
import { Router } from 'express';
import { askRAGStream } from '../rag-stream.js';
import { searchDocuments } from '../search.js';

router.post('/stream', async (req, res) => {
    const { message } = req.body;
    
    // Set headers for Server-Sent Events (SSE)
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    
    try {
        // Get context first
        const sources = await searchDocuments(message);
        const context = sources.map(s => s.content).join('\\n\\n');
        
        // Stream the response
        for await (const chunk of askRAGStream(message, context)) {
            res.write(\\`data: \\${JSON.stringify({ text: chunk })}\\n\\n\\`);
        }
        
        // Send sources at the end
        res.write(\\`data: \\${JSON.stringify({ sources, done: true })}\\n\\n\\`);
        res.end();
    } catch (error) {
        res.write(\\`data: \\${JSON.stringify({ error: 'Stream failed' })}\\n\\n\\`);
        res.end();
    }
});</code></pre>
        
        <h3>Frontend: Consuming the Stream</h3>
        <pre><code>// Frontend JavaScript
async function askQuestion(question) {
    const responseDiv = document.getElementById('response');
    responseDiv.textContent = '';
    
    const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const text = decoder.decode(value);
        const lines = text.split('\\n').filter(line => line.startsWith('data: '));
        
        for (const line of lines) {
            const data = JSON.parse(line.slice(6));
            if (data.text) {
                responseDiv.textContent += data.text;
            }
            if (data.sources) {
                showSources(data.sources);
            }
        }
    }
}</code></pre>
    """
    content += build_section("content", "Streaming Responses", streaming_section, "🌊")
    
    # Error Handling
    error_section = """
        <h3>Robust Error Handling</h3>
        <p>Production apps need graceful failure handling:</p>
        
        <pre><code>// src/rag-safe.ts
import OpenAI from 'openai';

const openai = new OpenAI();

interface RAGResult {
    success: boolean;
    answer?: string;
    error?: string;
    fallback?: boolean;
}

export async function askRAGSafe(question: string): Promise&lt;RAGResult&gt; {
    try {
        // Try to get context
        let context = '';
        try {
            const sources = await searchDocuments(question);
            context = sources.map(s => s.content).join('\\n\\n');
        } catch (dbError) {
            console.error('Database error:', dbError);
            // Continue without context - will use LLM knowledge
        }
        
        // Generate response with retry
        const response = await retryWithBackoff(async () => {
            return openai.chat.completions.create({
                model: 'gpt-4o-mini',
                messages: [
                    { 
                        role: 'system', 
                        content: context 
                            ? \\`Answer based on: \\${context}\\`
                            : 'You are a helpful assistant.'
                    },
                    { role: 'user', content: question }
                ],
                timeout: 30000  // 30 second timeout
            });
        }, 3);  // Retry up to 3 times
        
        return {
            success: true,
            answer: response.choices[0].message.content,
            fallback: !context  // Flag if we had to use fallback
        };
        
    } catch (error) {
        console.error('RAG failed:', error);
        
        // Return user-friendly error
        return {
            success: false,
            error: 'I\\'m having trouble answering right now. Please try again.'
        };
    }
}

// Retry helper with exponential backoff
async function retryWithBackoff&lt;T&gt;(
    fn: () => Promise&lt;T&gt;,
    maxRetries: number,
    baseDelay: number = 1000
): Promise&lt;T&gt; {
    let lastError: Error;
    
    for (let i = 0; i &lt; maxRetries; i++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error as Error;
            
            // Don't retry on certain errors
            if (error.status === 401 || error.status === 403) {
                throw error;  // Auth errors won't fix with retry
            }
            
            // Wait before retry (exponential backoff)
            const delay = baseDelay * Math.pow(2, i);
            console.log(\\`Retry \\${i + 1}/\\${maxRetries} after \\${delay}ms\\`);
            await new Promise(r => setTimeout(r, delay));
        }
    }
    
    throw lastError!;
}</code></pre>
        
        <h3>Rate Limiting</h3>
        <pre><code>// src/middleware/rateLimit.ts
import rateLimit from 'express-rate-limit';

export const chatLimiter = rateLimit({
    windowMs: 60 * 1000,  // 1 minute
    max: 20,              // 20 requests per minute
    message: { 
        error: 'Too many requests. Please wait a moment.' 
    },
    standardHeaders: true,
    legacyHeaders: false
});

// Apply to routes
app.use('/api/chat', chatLimiter);</code></pre>
    """
    content += build_section("content", "Error Handling & Resilience", error_section, "🛡️")
    
    # Logging & Monitoring
    logging_section = """
        <h3>Structured Logging</h3>
        <p>Good logs are essential for debugging production issues:</p>
        
        <pre><code>// src/logger.ts
import winston from 'winston';

export const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: 'logs/app.log' })
    ]
});

// Log every RAG request
export function logRAGRequest(data: {
    question: string;
    chunksFound: number;
    tokensUsed: number;
    responseTime: number;
    success: boolean;
}) {
    logger.info('RAG Request', {
        type: 'rag_request',
        ...data
    });
}</code></pre>
        
        <h3>Request Tracking</h3>
        <pre><code>// src/middleware/requestLogger.ts
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../logger.js';

export function requestLogger(req, res, next) {
    const requestId = uuidv4();
    const startTime = Date.now();
    
    // Attach to request for use in handlers
    req.requestId = requestId;
    
    // Log when response finishes
    res.on('finish', () => {
        logger.info('HTTP Request', {
            requestId,
            method: req.method,
            path: req.path,
            statusCode: res.statusCode,
            duration: Date.now() - startTime,
            userAgent: req.get('user-agent')
        });
    });
    
    next();
}</code></pre>
        
        <h3>Metrics to Track</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Metric</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Why It Matters</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Response Time</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">User experience, SLA compliance</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Token Usage</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Cost control</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Chunks Retrieved</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Search quality indicator</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Error Rate</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">System health</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">No-Result Queries</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Content gaps in your knowledge base</td>
            </tr>
        </table>
    """
    content += build_section("content", "Logging & Monitoring", logging_section, "📊")
    
    # Cost Optimization
    cost_section = """
        <h3>Understanding Costs</h3>
        <p>AI API costs can surprise you. Here's how to keep them under control:</p>
        
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Component</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Cost Driver</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Optimization</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Embeddings</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">$0.02/1M tokens</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Cache query embeddings</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">GPT-4o-mini</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">$0.15/1M input tokens</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Limit context size</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">GPT-4o</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">$2.50/1M input tokens</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Use only when needed</td>
            </tr>
        </table>
        
        <h3>Caching Strategies</h3>
        <pre><code>// src/cache.ts
import NodeCache from 'node-cache';

// Cache embeddings for 1 hour
const embeddingCache = new NodeCache({ stdTTL: 3600 });

export async function getCachedEmbedding(text: string): Promise&lt;number[]&gt; {
    const cacheKey = \\`emb:\\${hashString(text)}\\`;
    
    // Check cache first
    const cached = embeddingCache.get&lt;number[]&gt;(cacheKey);
    if (cached) {
        return cached;
    }
    
    // Generate and cache
    const embedding = await generateEmbedding(text);
    embeddingCache.set(cacheKey, embedding);
    
    return embedding;
}

// Cache frequent Q&A responses
const responseCache = new NodeCache({ stdTTL: 1800 }); // 30 min

export async function getCachedResponse(question: string): Promise&lt;string | null&gt; {
    const normalized = question.toLowerCase().trim();
    const cacheKey = \\`resp:\\${hashString(normalized)}\\`;
    
    return responseCache.get(cacheKey) || null;
}</code></pre>
        
        <h3>Model Selection Strategy</h3>
        <pre><code>// Use cheaper model for simple questions
function selectModel(question: string, context: string): string {
    const totalTokens = estimateTokens(question + context);
    
    // Simple questions with limited context → cheap model
    if (totalTokens &lt; 500 && !needsReasoning(question)) {
        return 'gpt-4o-mini';
    }
    
    // Complex questions or lots of context → better model
    return 'gpt-4o';
}

function needsReasoning(question: string): boolean {
    const reasoningKeywords = [
        'compare', 'analyze', 'explain why', 'difference between',
        'pros and cons', 'best approach', 'recommend'
    ];
    return reasoningKeywords.some(kw => 
        question.toLowerCase().includes(kw)
    );
}</code></pre>
    """
    content += build_section("content", "Cost Optimization", cost_section, "💰")
    
    # Deployment
    deploy_section = """
        <h3>Deployment Options</h3>
        
        <h4>Option 1: Docker Deployment</h4>
        <pre><code># Dockerfile
FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY dist/ ./dist/

ENV NODE_ENV=production
EXPOSE 3000

CMD ["node", "dist/index.js"]</code></pre>
        
        <pre><code># docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/rag
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
  
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: rag
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:</code></pre>
        
        <h4>Option 2: Cloud Platforms</h4>
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Platform</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Best For</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Managed DB Option</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Railway</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Quick deploys, free tier</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">PostgreSQL included</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Render</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Auto-scaling, simple</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">PostgreSQL included</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Fly.io</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Edge deployment</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Postgres + pgvector</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">AWS/GCP/Azure</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Enterprise, full control</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">RDS/Cloud SQL</td>
            </tr>
        </table>
        
        <h3>Environment Variables</h3>
        <pre><code># .env.production
NODE_ENV=production
PORT=3000

# Database
DATABASE_URL=postgresql://user:pass@host:5432/rag?sslmode=require

# OpenAI
OPENAI_API_KEY=sk-...

# Security
API_KEY_REQUIRED=true
ALLOWED_ORIGINS=https://yourdomain.com

# Limits
MAX_TOKENS=1000
RATE_LIMIT_PER_MINUTE=20</code></pre>
    """
    content += build_section("content", "Deployment", deploy_section, "🚀")
    
    # Security
    security_section = """
        <h3>Security Checklist</h3>
        
        <div class="callout callout-warning">
            <strong>⚠️ Before Going Live:</strong>
            <ul style="margin: 0.5rem 0 0 0;">
                <li>Never expose your OpenAI API key to the frontend</li>
                <li>Always validate and sanitize user input</li>
                <li>Implement rate limiting</li>
                <li>Use HTTPS in production</li>
                <li>Set up proper CORS policies</li>
            </ul>
        </div>
        
        <h3>Input Validation</h3>
        <pre><code>// src/middleware/validation.ts
import { z } from 'zod';

const chatSchema = z.object({
    message: z.string()
        .min(1, 'Message required')
        .max(2000, 'Message too long')
        .transform(s => s.trim()),
    conversationId: z.string().uuid().optional()
});

export function validateChat(req, res, next) {
    const result = chatSchema.safeParse(req.body);
    
    if (!result.success) {
        return res.status(400).json({
            error: 'Invalid request',
            details: result.error.issues
        });
    }
    
    req.body = result.data;
    next();
}</code></pre>
        
        <h3>Prompt Injection Prevention</h3>
        <pre><code>// Basic prompt injection detection
function detectPromptInjection(input: string): boolean {
    const suspiciousPatterns = [
        /ignore (previous|above|all) instructions/i,
        /disregard (your|the) (rules|instructions)/i,
        /you are now/i,
        /new instruction/i,
        /system prompt/i
    ];
    
    return suspiciousPatterns.some(p => p.test(input));
}

// Use in your RAG handler
if (detectPromptInjection(question)) {
    logger.warn('Potential prompt injection', { question });
    return { error: 'Invalid question format' };
}</code></pre>
    """
    content += build_section("content", "Security Considerations", security_section, "🔒")
    
    # Final Checklist
    checklist = """
        <h3>Production Readiness Checklist</h3>
        
        <div style="background: #161b22; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="margin-top: 0;">✅ Core Functionality</h4>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li>☑️ RAG pipeline working end-to-end</li>
                <li>☑️ Streaming responses implemented</li>
                <li>☑️ Conversation memory working</li>
                <li>☑️ Source citations included</li>
            </ul>
            
            <h4>✅ Reliability</h4>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li>☑️ Error handling with retries</li>
                <li>☑️ Graceful degradation</li>
                <li>☑️ Rate limiting configured</li>
                <li>☑️ Timeouts set</li>
            </ul>
            
            <h4>✅ Observability</h4>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li>☑️ Structured logging</li>
                <li>☑️ Request tracking</li>
                <li>☑️ Error alerting</li>
                <li>☑️ Cost monitoring</li>
            </ul>
            
            <h4>✅ Security</h4>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li>☑️ API keys secured</li>
                <li>☑️ Input validation</li>
                <li>☑️ CORS configured</li>
                <li>☑️ HTTPS enabled</li>
            </ul>
        </div>
        
        <div class="callout callout-tip">
            <strong>🎉 Congratulations!</strong> You've completed the GenAI RAG Developer Course! You now have the knowledge to build, deploy, and maintain production-ready RAG applications.
        </div>
    """
    content += build_section("content", "Production Checklist", checklist, "📋")
    
    # Quiz
    quiz = """
        <div class="quiz-question">
            <p>1. Why is streaming important for RAG applications?</p>
            <ul class="quiz-options">
                <li>A) It reduces API costs</li>
                <li>B) It improves perceived response time for users</li>
                <li>C) It increases accuracy</li>
                <li>D) It uses less memory</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>2. What should you do when the database is temporarily unavailable?</p>
            <ul class="quiz-options">
                <li>A) Return an error immediately</li>
                <li>B) Crash the server</li>
                <li>C) Retry with backoff, then gracefully degrade</li>
                <li>D) Wait indefinitely</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>3. Which is the most cost-effective strategy for embeddings?</p>
            <ul class="quiz-options">
                <li>A) Generate fresh embeddings for every request</li>
                <li>B) Cache embeddings for frequently asked questions</li>
                <li>C) Use the largest embedding model available</li>
                <li>D) Skip embeddings entirely</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>4. What is prompt injection?</p>
            <ul class="quiz-options">
                <li>A) A way to make prompts faster</li>
                <li>B) An attack where users try to override system instructions</li>
                <li>C) A method to improve response quality</li>
                <li>D) A caching strategy</li>
            </ul>
        </div>
        
        <p style="margin-top: 1rem; color: #8b949e;"><em>Answers: 1-B, 2-C, 3-B, 4-B</em></p>
    """
    content += build_section("quiz", "Final Assessment", quiz, "✅")
    
    # References
    refs = """
        <ul>
            <li>
                <a href="https://platform.openai.com/docs/api-reference/streaming" target="_blank">
                    OpenAI Streaming Guide
                </a>
                <span style="color: #8b949e;"> - Official streaming documentation</span>
            </li>
            <li>
                <a href="https://www.docker.com/get-started/" target="_blank">
                    Docker Getting Started
                </a>
                <span style="color: #8b949e;"> - Container deployment basics</span>
            </li>
            <li>
                <a href="https://owasp.org/www-project-top-ten/" target="_blank">
                    OWASP Top 10
                </a>
                <span style="color: #8b949e;"> - Web security best practices</span>
            </li>
            <li>
                <a href="https://railway.app/docs" target="_blank">
                    Railway Documentation
                </a>
                <span style="color: #8b949e;"> - Easy cloud deployment</span>
            </li>
            <li>
                <a href="https://github.com/OWASP/CheatSheetSeries/blob/master/cheatsheets/LLM_AI_Security_Cheat_Sheet.md" target="_blank">
                    OWASP LLM Security
                </a>
                <span style="color: #8b949e;"> - AI-specific security guidance</span>
            </li>
        </ul>
    """
    content += build_section("references", "Sources & Further Reading", refs, "📚")
    
    # Generate HTML
    html = MODULE_TEMPLATE.format(
        title="Module 6: Production & Polish",
        description="Take your RAG app from prototype to production",
        time_estimate="~90 minutes",
        difficulty="Advanced",
        module_number="6",
        content=content,
        prev_link='<a href="../module-5-rag/index.html">← Previous: RAG Pipeline</a>',
        next_link='<a href="../module-7-advanced/index.html">Next: Advanced RAG Techniques →</a>'
    )
    
    # Save
    output_dir = os.path.join(COURSE_DIR, "module-6-production")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
    
    print("✅ Generated Module 6: Production & Polish")


def build_module_7():
    """Build Module 7: Advanced RAG Techniques"""
    
    content = ""
    
    # Objectives
    objectives = """
        <ul>
            <li>Implement Hybrid Search (Keyword + Vector) for better retrieval</li>
            <li>Use Re-ranking to improve search precision</li>
            <li>Evaluate your RAG pipeline using RAGAS metrics</li>
            <li>Understand advanced retrieval patterns</li>
        </ul>
    """
    content += build_section("objectives", "What You'll Learn", objectives, "🎯")
    
    # Prerequisites
    prereqs = """
        <p>Before starting this module, you should have:</p>
        <ul>
            <li>Completed Module 6: Production RAG</li>
            <li>A working RAG pipeline</li>
        </ul>
        <div class="callout callout-info">
            <strong>Level Up:</strong> These techniques separate "toy" demos from high-quality enterprise search systems.
        </div>
    """
    content += build_section("prerequisites", "Prerequisites", prereqs, "📋")
    
    # Hybrid Search
    hybrid_section = """
        <h3>The Problem with Vector Search</h3>
        <p>Vector search is amazing at understanding <em>meaning</em>, but sometimes fails at <em>exact matches</em>.</p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
            <div style="background: #21262d; padding: 1rem; border-radius: 8px;">
                <strong>Vector Search Wins 🧠</strong>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">Query: "How to fix connection issues"<br>Matches: "Troubleshooting network errors"</p>
            </div>
            <div style="background: #21262d; padding: 1rem; border-radius: 8px;">
                <strong>Vector Search Fails ❌</strong>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">Query: "Error code 0x80042"<br>Matches: Generic error pages (misses exact code)</p>
            </div>
        </div>
        
        <h3>Solution: Hybrid Search</h3>
        <p>Combine <strong>Vector Search</strong> (Semantic) with <strong>Keyword Search</strong> (BM25/Full-Text) using Reciprocal Rank Fusion (RRF).</p>
        
        <pre><code>-- Enable pg_trgm for better keyword search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Add a generated column for full-text search
ALTER TABLE documents 
ADD COLUMN tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('english', content)
) STORED;

CREATE INDEX ts_idx ON documents USING GIN (tsv);</code></pre>
        
        <h3>Implementing Hybrid Search</h3>
        <pre><code>// src/hybrid-search.ts
export async function hybridSearch(query: string, limit = 5) {
    // 1. Get Vector Results
    const vectorResults = await searchVector(query, limit * 2);
    
    // 2. Get Keyword Results
    const keywordResults = await searchKeyword(query, limit * 2);
    
    // 3. Combine with RRF (Reciprocal Rank Fusion)
    const combined = new Map&lt;number, number&gt;(); // id -> score
    const k = 60; // RRF constant
    
    // Score vector results
    vectorResults.forEach((doc, rank) => {
        const score = 1 / (k + rank + 1);
        combined.set(doc.id, (combined.get(doc.id) || 0) + score);
    });
    
    // Score keyword results
    keywordResults.forEach((doc, rank) => {
        const score = 1 / (k + rank + 1);
        combined.set(doc.id, (combined.get(doc.id) || 0) + score);
    });
    
    // Sort by final score
    return Array.from(combined.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, limit)
        .map(([id]) => getDocById(id));
}</code></pre>
    """
    content += build_section("content", "Hybrid Search", hybrid_section, "🔍")
    
    # Re-ranking
    rerank_section = """
        <h3>Why Re-ranking?</h3>
        <p>Vector search is fast but "lossy". It compresses text into fixed numbers. A <strong>Cross-Encoder</strong> (Re-ranker) looks at the full query and document pair to give a precise relevance score.</p>
        
        <div style="background: #161b22; padding: 1.5rem; border-radius: 8px; margin: 1rem 0; text-align: center;">
            <p style="font-family: monospace; margin: 0;">
                Retrieve 50 docs (Fast) → Re-rank top 50 (Slow but Precise) → Return top 5
            </p>
        </div>
        
        <h3>Using Cohere Re-rank</h3>
        <pre><code>import { CohereClient } from "cohere-ai";

const cohere = new CohereClient({ token: "YOUR_API_KEY" });

export async function rerankDocs(query: string, documents: string[]) {
    const rerank = await cohere.rerank({
        documents: documents,
        query: query,
        topN: 5,
        model: "rerank-english-v3.0"
    });
    
    return rerank.results.map(r => ({
        content: documents[r.index],
        score: r.relevanceScore
    }));
}</code></pre>
        
        <div class="callout callout-tip">
            <strong>Cost/Benefit:</strong> Re-ranking adds latency (~100-500ms) and cost, but can drastically improve answer quality by ensuring the LLM sees the absolute best context.
        </div>
    """
    content += build_section("content", "Re-ranking", rerank_section, "📊")
    
    # Evaluation
    eval_section = """
        <h3>How Do You Know It's Good?</h3>
        <p>You can't improve what you don't measure. <strong>RAGAS</strong> (Retrieval Augmented Generation Assessment) is the standard framework for evaluating RAG.</p>
        
        <h3>Key Metrics</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <tr style="background: #161b22;">
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Metric</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Question</th>
                <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d;">Measures</th>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><strong>Faithfulness</strong></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">"Is the answer derived <em>only</em> from the context?"</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Hallucinations</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><strong>Answer Relevance</strong></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">"Does the answer actually address the user's query?"</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Response Quality</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;"><strong>Context Precision</strong></td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">"Did we retrieve relevant documents?"</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #30363d;">Retrieval Quality</td>
            </tr>
        </table>
        
        <h3>Implementing Evaluation</h3>
        <p>You can use an LLM to evaluate your RAG system!</p>
        
        <pre><code>// Simple Faithfulness Check
async function checkFaithfulness(context: string, answer: string) {
    const prompt = `
    Given the context: "${context}"
    And the answer: "${answer}"
    
    Rate from 0.0 to 1.0: How much of the answer is supported by the context?
    Return only the number.
    `;
    
    const score = await askLLM(prompt);
    return parseFloat(score);
}</code></pre>
    """
    content += build_section("content", "Evaluation (RAGAS)", eval_section, "📏")
    
    # Quiz
    quiz = """
        <div class="quiz-question">
            <p>1. When should you use Hybrid Search?</p>
            <ul class="quiz-options">
                <li>A) Always, it's free</li>
                <li>B) When users search for specific IDs, codes, or exact names</li>
                <li>C) When you want to save database space</li>
                <li>D) When you don't have embeddings</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>2. What is the main trade-off of Re-ranking?</p>
            <ul class="quiz-options">
                <li>A) Lower accuracy</li>
                <li>B) Higher latency and cost</li>
                <li>C) More complex code</li>
                <li>D) Requires more storage</li>
            </ul>
        </div>
        
        <div class="quiz-question">
            <p>3. What does "Faithfulness" measure in RAG evaluation?</p>
            <ul class="quiz-options">
                <li>A) How polite the bot is</li>
                <li>B) If the answer is grounded in the retrieved context</li>
                <li>C) How fast the response was</li>
                <li>D) If the user liked the answer</li>
            </ul>
        </div>
        
        <p style="margin-top: 1rem; color: #8b949e;"><em>Answers: 1-B, 2-B, 3-B</em></p>
    """
    content += build_section("quiz", "Check Your Understanding", quiz, "✅")
    
    # References
    refs = """
        <ul>
            <li>
                <a href="https://docs.ragas.io/en/latest/" target="_blank">
                    RAGAS Documentation
                </a>
                <span style="color: #8b949e;"> - The standard for RAG evaluation</span>
            </li>
            <li>
                <a href="https://txt.cohere.com/rerank/" target="_blank">
                    Cohere Re-rank Blog
                </a>
                <span style="color: #8b949e;"> - Deep dive into re-ranking</span>
            </li>
            <li>
                <a href="https://www.pinecone.io/learn/hybrid-search-intro/" target="_blank">
                    Pinecone: Hybrid Search
                </a>
                <span style="color: #8b949e;"> - Understanding sparse + dense vectors</span>
            </li>
        </ul>
    """
    content += build_section("references", "Sources & Further Reading", refs, "📚")
    
    # Generate HTML
    html = MODULE_TEMPLATE.format(
        title="Module 7: Advanced RAG Techniques",
        description="Master Hybrid Search, Re-ranking, and Evaluation",
        time_estimate="~60 minutes",
        difficulty="Expert",
        module_number="7",
        content=content,
        prev_link='<a href="../module-6-production/index.html">← Previous: Production RAG</a>',
        next_link='<span style="color: #3fb950;">🎉 Course Complete!</span>'
    )
    
    # Save
    output_dir = os.path.join(COURSE_DIR, "module-7-advanced")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
    
    print("✅ Generated Module 7: Advanced RAG Techniques")


def main():
    print("Generating course modules...")
    print("="*60)
    
    build_module_0()
    build_module_1()
    build_module_2()
    build_module_3()
    build_module_4()
    build_module_5()
    build_module_6()
    build_module_7()
    
    print("="*60)
    print("Done! Check data/course/ for generated modules.")

if __name__ == "__main__":
    main()
