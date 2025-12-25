# GenAI / RAG Developer Learning Path

**Your background:** Angular + Node developer  
**Target role:** Applied GenAI Engineer / RAG Developer  
**Timeline:** 10–14 weeks (5–7 hrs/week)  
**Stack:** Express + TypeScript + Postgres + pgvector + OpenAI

---

## Prerequisites & Background Check

### ✅ What You Should Already Have

| Skill | Level Needed | You Have It? |
|-------|--------------|--------------|
| **JavaScript** | Comfortable with async/await, promises, array methods | ✅ (Angular dev) |
| **TypeScript** | Basic types, interfaces, generics | ✅ (Angular uses TS) |
| **Node.js** | Can build REST APIs, use npm | ✅ (Node dev) |
| **HTTP/REST** | Understand GET/POST, status codes, JSON | ✅ |
| **Git** | Clone, commit, push, branches | ✅ |
| **Terminal** | Navigate folders, run commands | ✅ |
| **VS Code** | Basic usage | ✅ |

> **Verdict:** You have all the prerequisites. No gaps to fill before starting.

---

### 📚 Things You'll Learn (Not Prerequisites)

| Topic | Covered In | Don't Learn Before |
|-------|------------|-------------------|
| **Postgres/SQL** | Phase 1 | ✅ Covered in Week 1 |
| **Docker basics** | Phase 1 | ✅ Just 3 commands needed |
| **LLM concepts** | Phase 2 | ✅ Explained from scratch |
| **Embeddings/Vectors** | Phase 3 | ✅ Math-free explanation |
| **RAG architecture** | Phase 3 | ✅ Step-by-step |
| **Python** | Phase 7 (optional) | ✅ Not required for this path |

---

### ❌ What You Do NOT Need

| Don't Need | Why |
|------------|-----|
| **Machine Learning knowledge** | You're using LLMs, not building them |
| **Python** | We use TypeScript (you can add Python later) |
| **Math (linear algebra, etc.)** | Not needed for applied GenAI work |
| **GPU/CUDA experience** | Using APIs + Ollama handles this |
| **Data Science background** | Different role entirely |
| **LangChain experience** | We build from scratch first, then optionally use frameworks |

---

### 🔧 Setup Before Week 1

**Install these (15 minutes total):**

```bash
# 1. Node.js (you probably have this)
node --version  # Should be 18+ 

# 2. Docker (for Postgres)
# - Linux: install Docker Engine + Docker Compose plugin
# - Windows/Mac: Docker Desktop
# Download: https://www.docker.com/products/docker-desktop/
docker --version

# 3. VS Code extensions
# - Prisma
# - ESLint
# - Thunder Client (API testing) or use Postman

# 4. OpenAI API key (free tier works)
# Get from: https://platform.openai.com/api-keys

# 5. (Optional) Ollama for local LLMs
curl -fsSL https://ollama.com/install.sh | sh
```

**Create project folder:**
```bash
mkdir -p ~/projects/rag-knowledge-chat
cd ~/projects/rag-knowledge-chat
npm init -y
```

---

## Week 1 Recipe (Follow Step-by-Step)

### Goal
- Local Postgres (pgvector) running in Docker
- Express + TypeScript API running
- Prisma connected + first migration applied

### Step 0 — Verify tools
```bash
node --version   # expect v18+ (or v20+)
git --version
docker --version
docker compose version
```

If `docker compose` is missing on Linux, install Docker Engine + Compose plugin (recommended) or Docker Desktop.

---

### Step 1 — Create repo + install deps
```bash
mkdir -p ~/projects/rag-knowledge-chat
cd ~/projects/rag-knowledge-chat

npm init -y
npm i express zod dotenv
npm i -D typescript ts-node-dev @types/node @types/express prisma
npx tsc --init
npx prisma init
```

Expected:
- `package.json` created
- `tsconfig.json` created
- `prisma/schema.prisma` created
- `.env` created (by Prisma)

---

### Step 2 — Start Postgres + pgvector (Docker)
Create `docker-compose.yml` using the template in this doc, then run:

```bash
docker compose up -d
docker compose ps
```

Expected: service `postgres` is `running`.

---

### Step 3 — Set your DATABASE_URL
In your `.env` (project root), set:

```bash
DATABASE_URL=postgresql://raguser:ragpass@localhost:5432/ragdb
```

Quick check (optional):
```bash
docker compose logs -n 50 postgres
```

Expected: Postgres ready to accept connections.

---

### Step 4 — Add Prisma schema + run first migration
Replace `prisma/schema.prisma` with the schema in this doc, then run:

```bash
npx prisma generate
npx prisma migrate dev --name init
```

Expected:
- Migration created in `prisma/migrations/...`
- Database tables created

---

### Step 5 — Create a minimal API (health check)

Create `src/index.ts`:

```ts
import 'dotenv/config';
import express from 'express';

const app = express();
app.use(express.json());

app.get('/health', (_req, res) => res.json({ ok: true }));

const port = Number(process.env.PORT ?? 3000);
app.listen(port, () => {
   console.log(`API listening on http://localhost:${port}`);
});
```

Add scripts in `package.json`:

```json
{
   "scripts": {
      "dev": "ts-node-dev --respawn --transpile-only src/index.ts"
   }
}
```

Run:
```bash
mkdir -p src
npm run dev
```

Expected:
- Terminal prints `API listening on http://localhost:3000`

Test:
```bash
curl -s http://localhost:3000/health
```

Expected output:
```json
{"ok":true}
```

---

### Step 6 — Week 1 done criteria
- Docker Postgres running (`docker compose ps` looks healthy)
- Prisma migrated successfully
- `/health` endpoint returns `{ ok: true }`

If you want, next I can generate the Week 2 recipe (URL ingestion + chunking) in the same style.

---

## 🧠 Pro Tips for Success (Read Before Starting)

### 1. The "Full Stack TypeScript" Advantage
Since you use **Express (Backend)** and **Angular (Frontend)**, you have a superpower: **Shared Types**.
- **Don't duplicate interfaces.** Create a `shared` folder (or use a monorepo/workspace) to define API response types (e.g., `ChatResponse`, `ErrorResponse`).
- **Why:** If you change the backend API, your Angular build will fail immediately. This saves hours of debugging "undefined" errors.

### 2. Treat "Git" Like a Feature
Recruiters look at commit history, not just the final code.
- **Don't** push one giant "Initial commit" at the end.
- **Do** commit every time you finish a task:
  - `feat: add pgvector migration`
  - `fix: improve chunking overlap`
  - `docs: add setup instructions`
- **Why:** It proves you work iteratively and professionally.

### 3. Avoid the "Prompt Engineering" Trap
Beginners spend 80% of time tweaking prompts ("Act as a helpful assistant...").
- **Don't** obsess over prompts early on.
- **Do** spend 80% of your time on **Retrieval** (getting the right chunks).
- **Why:** If retrieval is bad (garbage in), the best prompt won't fix it (garbage out). Focus on **Phase 3.3 (Chunking)** and **Phase 3.5 (Vector Search)**—that's where the real engineering happens.

---

## Phase 1: Foundation (Weeks 1–2)
### Goal: Get back to coding + learn Postgres properly

### 1.1 Postgres Fundamentals (MUST LEARN)
**Why:** Every GenAI app needs: users, documents, chunks, chat history, permissions, logs.

**Topics to learn:**
- Tables, primary keys, foreign keys, constraints
- Data types (TEXT, JSONB, TIMESTAMP, UUID)
- Relationships: one-to-many, many-to-many
- Joins: INNER, LEFT, RIGHT
- Indexes: B-tree, when to use them
- EXPLAIN ANALYZE (understand query performance)
- Migrations (schema versioning)

**Resources (free):**
1. **PostgreSQL Tutorial** (official-style)  
   https://www.postgresqltutorial.com/  
   → Do: Basic, Joins, Indexes sections

2. **SQLBolt** (interactive)  
   https://sqlbolt.com/  
   → Lessons 1–12 (1–2 hours total)

3. **Prisma Docs** (ORM you'll use)  
   https://www.prisma.io/docs/getting-started  
   → Quickstart + Data Modeling sections

**Practice (30 min):**
- Create a `users` table with email, password_hash, created_at
- Create an `orgs` table
- Create `memberships` (user ↔ org many-to-many)
- Write a query: "all orgs a user belongs to" (JOIN)

---

### 1.2 Node.js + TypeScript Refresh
**Why:** You know this, but need hands-on again.

**Topics to review:**
- Project setup with TypeScript
- Express routing + middleware
- Request validation (Zod)
- Error handling patterns
- Environment variables (.env)

**Resources:**
1. **TypeScript Handbook** (skim if you know it)  
   https://www.typescriptlang.org/docs/handbook/intro.html

2. **Express + TypeScript setup**  
   https://blog.logrocket.com/how-to-set-up-node-typescript-express/

3. **Zod (validation)**  
   https://zod.dev/  
   → Just the "Basic usage" section

---

### 1.3 Docker Basics (for Postgres)
**Why:** Run Postgres locally without installing it system-wide.

**Topics:**
- What is Docker / containers (concept only)
- docker-compose.yml basics
- Starting/stopping containers

**Resources:**
1. **Docker in 100 Seconds** (video)  
   https://www.youtube.com/watch?v=Gjnup-PuquQ

2. **Docker Compose Quickstart**  
   https://docs.docker.com/compose/gettingstarted/

**You only need to know:**
```bash
# Modern Docker (recommended)
docker compose up -d    # start
docker compose down     # stop
docker compose logs     # see logs

# If your system only has the legacy binary, these also work:
# docker-compose up -d
# docker-compose down
# docker-compose logs
```

---

## Phase 2: LLM Fundamentals (Week 3)
### Goal: Understand how to call LLMs properly

### 2.1 Core Concepts
**Topics:**
- What is a token? (word pieces)
- Context window (max input size)
- Temperature (randomness)
- System prompt vs user prompt
- Streaming responses
- Structured outputs (JSON mode)
- Function/tool calling

**Resources:**
1. **OpenAI API Docs** (essential)  
   https://platform.openai.com/docs/introduction  
   → Read: Models, Chat Completions, Structured Outputs

2. **Prompt Engineering Guide** (practical)  
   https://www.promptingguide.ai/  
   → Read: Introduction, Techniques (first 3–4)

3. **OpenAI Cookbook** (examples)  
   https://cookbook.openai.com/  
   → Look at: "How to format inputs" and "How to stream completions"

**Practice (1 hour):**
- Get an OpenAI API key (free tier or pay-as-you-go)
- Write a simple Node script that:
  - Sends a prompt
  - Gets a response
  - Streams the response to console

---

### 2.2 Structured Outputs + Tool Calling
**Why:** Real apps need JSON responses, not random text.

**Topics:**
- JSON mode / response_format
- Zod schema → OpenAI structured output
- Function calling basics

**Resources:**
1. **OpenAI Structured Outputs**  
   https://platform.openai.com/docs/guides/structured-outputs

2. **OpenAI Function Calling**  
   https://platform.openai.com/docs/guides/function-calling

**Practice:**
- Make the LLM return: `{ "answer": "...", "confidence": "high/medium/low" }`
- Validate the response with Zod

---

## Phase 3: RAG Fundamentals (Weeks 4–6)
### Goal: Build a working RAG pipeline

### 3.1 What is RAG?
**Concept:**
- LLMs don't know your private data
- RAG = Retrieval-Augmented Generation
- Steps: Query → Retrieve relevant chunks → Add to prompt → Generate answer

**Resources:**
1. **RAG Explained (video, 10 min)**  
   https://www.youtube.com/watch?v=T-D1OfcDW1M

2. **LangChain RAG Tutorial** (concept, not code)  
   https://python.langchain.com/docs/tutorials/rag/  
   → Read for understanding (you'll implement in TS)

3. **Pinecone: What is RAG?**  
   https://www.pinecone.io/learn/retrieval-augmented-generation/

---

### 3.2 Document Ingestion
**Topics:**
- Loading documents (URL, Markdown, PDF later)
- Text extraction
- Cleaning / preprocessing

**Resources:**
1. **Cheerio** (HTML parsing in Node)  
   https://cheerio.js.org/

2. **Mozilla Readability** (extract article text)  
   https://github.com/mozilla/readability

**Practice:**
- Fetch a URL
- Extract main text content
- Store in database

---

### 3.3 Chunking
**Topics:**
- Why chunk? (context window limits, better retrieval)
- Chunk size (typically 500–1500 chars)
- Overlap (50–200 chars)
- Metadata (source, page, section)

**Resources:**
1. **Chunking Strategies** (article)  
   https://www.pinecone.io/learn/chunking-strategies/

2. **LangChain Text Splitters** (concept reference)  
   https://js.langchain.com/docs/concepts/text_splitters

**Practice:**
- Implement a simple splitter: split by paragraphs, then by size
- Add metadata: { documentId, chunkIndex, charStart, charEnd }

---

### 3.4 Embeddings
**Topics:**
- What is an embedding? (text → vector of numbers)
- Semantic similarity (cosine distance)
- OpenAI embeddings API

**Resources:**
1. **OpenAI Embeddings Guide**  
   https://platform.openai.com/docs/guides/embeddings

2. **What are Embeddings? (video)**  
   https://www.youtube.com/watch?v=wjZofJX0v4M

**Practice:**
- Call OpenAI embeddings API
- Store vectors in pgvector
- Query: find top 5 similar chunks to a question

---

### 3.5 Vector Database (pgvector)
**Topics:**
- What is a vector database?
- pgvector: Postgres extension for vectors
- Indexing vectors (IVFFlat, HNSW)
- Similarity search queries

**Resources:**
1. **pgvector GitHub** (setup)  
   https://github.com/pgvector/pgvector

2. **pgvector + Prisma**  
   https://www.prisma.io/docs/orm/prisma-schema/data-model/unsupported-database-features#scalars

3. **Supabase pgvector Guide** (good explanation)  
   https://supabase.com/docs/guides/ai/vector-columns

**Practice:**
- Add vector column to chunks table
- Store embeddings
- Write similarity search query

---

### 3.6 Retrieval + Generation
**Topics:**
- Query embedding → retrieve top-k → build prompt → generate
- Citation patterns (show sources)
- "Refuse if no relevant context"

**Resources:**
1. **RAG Prompt Patterns**  
   https://www.promptingguide.ai/research/rag

**Practice:**
- Build `/chat` endpoint:
  1. Embed the question
  2. Retrieve top 5 chunks
  3. Build prompt: "Answer using ONLY these sources: ..."
  4. Call LLM
  5. Return answer + chunk IDs as citations

---

## Phase 4: Reliability + Evaluation (Weeks 7–8)
### Goal: Make your RAG app production-quality

### 4.1 Why Evaluation Matters
- LLM apps fail silently (wrong answers look confident)
- You need automated checks before deploying changes

**Topics:**
- Test sets (question + expected behavior)
- Metrics: answer quality, groundedness, refusal accuracy
- Regression testing

**Resources:**
1. **LLM Evaluation Guide**  
   https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation

2. **RAGAS** (RAG evaluation framework, Python but concepts apply)  
   https://docs.ragas.io/en/latest/concepts/metrics/

**Practice:**
- Create `eval/questions.json` with 20–50 Q&A pairs
- Write a script that runs each question and checks:
  - Answer is not empty
  - Citations exist
  - If no context, model refuses

---

### 4.2 Monitoring + Observability
**Topics:**
- Logging: query, chunks retrieved, latency, tokens used
- Cost tracking (tokens × price)
- User feedback (thumbs up/down)

**Practice:**
- Add a `logs` table: timestamp, user_id, query, answer, latency_ms, token_count
- Add `/feedback` endpoint: message_id, rating, comment

---

## Phase 5: Security + Safety (Week 9)
### Goal: Ship safely

### 5.1 Prompt Injection
**What:** Malicious text in documents can hijack LLM behavior.

**Resources:**
1. **OWASP LLM Top 10**  
   https://owasp.org/www-project-top-10-for-large-language-model-applications/

2. **Prompt Injection Explained**  
   https://simonwillison.net/2022/Sep/12/prompt-injection/

**Mitigations:**
- Treat retrieved chunks as untrusted data
- Use structured outputs (harder to hijack)
- Validate LLM output before returning

---

### 5.2 Access Control (RBAC)
**Topics:**
- Users can only retrieve chunks from docs they have access to
- Filter by org_id / user_id in vector search

**Practice:**
- Add `org_id` to documents and chunks
- Filter retrieval: `WHERE org_id = $user_org_id`

---

### 5.3 PII + Data Handling
**Topics:**
- Don't log sensitive user data
- Mask or redact PII before sending to LLM (if needed)
- Data retention policies

---

## Phase 6: Deployment (Week 10)
### Goal: Ship publicly

### 6.1 Deployment Options (simple)
- **Render** (easiest): https://render.com/
- **Railway**: https://railway.app/
- **Fly.io**: https://fly.io/

**Practice:**
- Deploy API to Render/Railway
- Use managed Postgres (or deploy yours)
- Set environment variables securely

---

### 6.2 README + Portfolio
**What to include:**
- Architecture diagram
- How RAG works in your app
- Security measures
- Evaluation results (e.g., "85% of test questions answered correctly with citations")
- Screenshots / demo link

---

## Phase 6.5: Angular Chat UI (Week 10–11)
### Goal: Add a frontend so you can demo the app

**Why:** Recruiters want to see a working demo, not just an API.

**Topics:**
- Simple chat interface (messages list + input)
- Streaming responses (SSE / fetch with reader)
- Show citations (clickable sources)
- Loading states + error handling

**Resources:**
1. **Angular SSE (Server-Sent Events)**  
   https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events

2. **Streaming fetch in JS**  
   https://developer.mozilla.org/en-US/docs/Web/API/Streams_API/Using_readable_streams

**Practice:**
- Create `/chat` page with:
  - Message history (user + assistant bubbles)
  - Input box + send button
  - Streaming text display
  - Citations shown below answer (click to open source)

---

## Phase 7 (Optional): Add Python Skills
### When: After you ship the TS project

**Why Python:**
- Many ML tools are Python-first
- Evaluation frameworks (RAGAS, DeepEval)
- Fine-tuning / experiments

**What to learn:**
- Python basics (if rusty)
- Jupyter notebooks
- pandas (data manipulation)
- Write eval scripts in Python

**Resources:**
1. **Python for JS Developers**  
   https://www.pythonforjavascriptdevelopers.com/

2. **Real Python** (tutorials)  
   https://realpython.com/

---

## Bonus: Advanced Topics (After MVP)

### Cost Management + Caching
**Why:** OpenAI API costs money. Caching saves 50–80% on repeated queries.

**Topics:**
- Token counting before sending
- Budget limits per user/org
- Cache embeddings (same text = same vector)
- Cache LLM responses for identical queries (with TTL)

**Practice:**
- Add Redis for caching (or in-memory for MVP)
- Log token usage per request
- Add admin endpoint: "cost this month"

---

### 🆓 Local LLMs (Zero Cost Learning)

**Why:** Learn RAG concepts without spending on API costs. Your laptop (32GB RAM, i7) can run 7B–13B models.

**Best tool: Ollama** (easiest setup, most popular)

**Install Ollama (one command):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

#### Understanding Model Sizes & Quantization

**What is quantization?**  
Models are compressed from 16-bit to 4-bit, reducing size by 75% with minimal quality loss.  
Ollama typically downloads **quantized GGUF** variants (often ~4-bit by default). The exact quantization can vary by model/tag, but the defaults are fine for learning.

**Memory rule of thumb:**
| Model Size | Download Size | RAM Needed | Your Laptop (32GB) |
|------------|---------------|------------|-------------------|
| 1B–3B | 0.5–2GB | 2–4GB | ✅ Very fast |
| 7B–8B | 4–5GB | 6–8GB | ✅ Runs well |
| 13B | 7–8GB | 10–12GB | ✅ Works |
| 32B | 18–20GB | 24–28GB | ⚠️ Slow but possible |
| 70B+ | 40GB+ | 48GB+ | ❌ Too big |

---

#### Recommended Models for Your Laptop (32GB RAM, CPU-only)

| Model | Command | RAM | Speed | Best For | Popularity |
|-------|---------|-----|-------|----------|------------|
| **Qwen2.5 7B** ⭐ | `ollama run qwen2.5:7b` | ~5GB | 5-8 tok/s | Best overall for RAG | 18.5M pulls |
| **Llama 3.2 3B** | `ollama run llama3.2:3b` | ~2GB | 10-15 tok/s | Fast iteration | 50.6M pulls |
| **Phi-3 Mini** | `ollama run phi3` | ~2.5GB | 10-12 tok/s | Reasoning, tiny | 15.3M pulls |
| **Mistral 7B** | `ollama run mistral` | ~4.5GB | 5-8 tok/s | Classic, reliable | 23.4M pulls |
| **Gemma2 9B** | `ollama run gemma2:9b` | ~6GB | 4-6 tok/s | Google quality | 11.9M pulls |
| **Qwen2.5-coder** | `ollama run qwen2.5-coder:7b` | ~5GB | 5-8 tok/s | Code generation | 9.4M pulls |
| **Llama3-ChatQA** | `ollama run llama3-chatqa` | ~5GB | 5-8 tok/s | RAG/QA specific | 165K pulls |

> **🎯 Start with:**  
> - `qwen2.5:7b` — Best quality, 128K context (great for RAG)  
> - `llama3.2:3b` — Fastest, use for quick tests

---

#### Quick Start Commands

```bash
# Get the best model for RAG (recommended)
ollama run qwen2.5:7b

# Or fastest model for quick tests
ollama run llama3.2:3b

# For coding tasks
ollama run qwen2.5-coder:7b

# Check running models
ollama list
```

**Use Ollama in your code (OpenAI-compatible API):**
```typescript
// Ollama exposes OpenAI-compatible API at localhost:11434
const response = await fetch('http://localhost:11434/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
      model: 'qwen2.5:7b',
    messages: [{ role: 'user', content: 'Hello!' }]
  })
});
```

**Or with OpenAI SDK (just change base URL):**
```typescript
import OpenAI from 'openai';

// For local development (free)
const localLLM = new OpenAI({
  baseURL: 'http://localhost:11434/v1',
  apiKey: 'not-needed'  // Ollama doesn't need a key
});

// For production (paid, better quality)
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Use same code, just switch client
const client = process.env.USE_LOCAL ? localLLM : openai;
```

**Local embeddings (also free):**
```bash
# Download embedding model
ollama pull nomic-embed-text
```
```typescript
const embedding = await fetch('http://localhost:11434/api/embeddings', {
  method: 'POST',
  body: JSON.stringify({
    model: 'nomic-embed-text',
    prompt: 'Your text here'
  })
});
```

**Recommended workflow:**
1. **Learning/dev:** Use Ollama (free, works offline)
2. **Testing quality:** Use OpenAI (if your account has free credits or pay-as-you-go)
3. **Production:** Use OpenAI or Anthropic (paid)

**Resources:**
- Ollama: https://ollama.com/
- Model library: https://ollama.com/library
- LM Studio (GUI alternative): https://lmstudio.ai/

---

### Reranking (improves retrieval quality)
**Why:** Vector search returns "similar" but not always "best". Reranking fixes this.

**Topics:**
- What is reranking? (re-score retrieved chunks)
- Cohere Rerank API (easy to add)
- Cross-encoder models (more advanced)

**Resources:**
1. **Cohere Rerank**  
   https://docs.cohere.com/docs/rerank

2. **Why Reranking Matters**  
   https://www.pinecone.io/learn/series/rag/rerankers/

**Practice:**
- After vector search, call rerank API
- Use top 3–5 reranked chunks for LLM

---

### Multi-turn Conversation (chat history)
**Why:** Real users ask follow-up questions. "What about X?" needs context.

**Topics:**
- Store conversation history in DB
- Include last N messages in prompt
- Manage context window limits

**Practice:**
- Add `chat_sessions` and `messages` tables
- `/chat` endpoint includes last 5 messages in prompt
- Handle "context too long" by summarizing or trimming

---

### Error Handling for LLM APIs
**Why:** APIs fail. Rate limits, timeouts, outages happen.

**Topics:**
- Retry with exponential backoff
- Timeout handling
- Graceful degradation ("Sorry, try again")
- Multiple provider fallback (OpenAI → Anthropic)

**Practice:**
- Wrap LLM calls in try/catch
- Add retry logic (max 3 attempts)
- Return friendly error to user

---

## Interview Preparation

### What interviewers ask for GenAI/RAG roles

**System design questions:**
- "Design a chat-with-docs system for 1000 users"
- "How would you handle 10GB of documents?"
- "How do you ensure answers are accurate?"

**Technical questions:**
- "Explain RAG step by step"
- "What's the difference between embeddings and fine-tuning?"
- "How do you prevent prompt injection?"
- "How do you evaluate a RAG system?"

**Behavioral / portfolio:**
- "Walk me through your project architecture"
- "What was the hardest bug you fixed?"
- "How did you decide chunk size?"

### How to talk about your project (template)

> "I built a RAG-based knowledge chat system where users can upload documents and ask questions with cited answers.
>
> **Architecture:** Express + TypeScript backend, Postgres with pgvector for embeddings, OpenAI for generation, Angular frontend with streaming.
>
> **Key features:** Role-based access (users only see their org's docs), citation links, feedback collection, automated evaluation suite.
>
> **Challenges I solved:**
> - Chunking strategy: tested 500/800/1200 chars, 800 worked best for our docs
> - Prompt injection: treated retrieved text as untrusted, used structured outputs
> - Evaluation: built 50-question test set, 87% answer accuracy
>
> **What I'd improve:** Add reranking, better PDF parsing, cost dashboard."

---

## Starter Code Templates

### docker-compose.yml (Postgres + pgvector)
```yaml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: rag-postgres
    environment:
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ragpass
      POSTGRES_DB: ragdb
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

### .env (example)
```
DATABASE_URL=postgresql://raguser:ragpass@localhost:5432/ragdb
OPENAI_API_KEY=sk-your-key-here
PORT=3000
```

### Prisma schema (starter)
```prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id           String       @id @default(uuid())
  email        String       @unique
  passwordHash String
  createdAt    DateTime     @default(now())
  memberships  Membership[]
  chatSessions ChatSession[]
}

model Org {
  id          String       @id @default(uuid())
  name        String
  createdAt   DateTime     @default(now())
  memberships Membership[]
  documents   Document[]
}

model Membership {
  id     String @id @default(uuid())
  userId String
  orgId  String
  role   String @default("member") // member, admin
  user   User   @relation(fields: [userId], references: [id])
  org    Org    @relation(fields: [orgId], references: [id])

  @@unique([userId, orgId])
}

model Document {
  id        String   @id @default(uuid())
  orgId     String
  title     String
  sourceUrl String?
  createdAt DateTime @default(now())
  org       Org      @relation(fields: [orgId], references: [id])
  chunks    Chunk[]
}

model Chunk {
  id         String   @id @default(uuid())
  documentId String
  content    String
  chunkIndex Int
  metadata   Json?
  // embedding added via raw SQL (pgvector)
  document   Document @relation(fields: [documentId], references: [id])

  @@index([documentId])
}

model ChatSession {
  id        String    @id @default(uuid())
  userId    String
  createdAt DateTime  @default(now())
  user      User      @relation(fields: [userId], references: [id])
  messages  Message[]
}

model Message {
  id            String      @id @default(uuid())
  sessionId     String
  role          String      // user, assistant
  content       String
  citations     Json?       // array of chunk IDs
  tokenCount    Int?
  latencyMs     Int?
  createdAt     DateTime    @default(now())
  session       ChatSession @relation(fields: [sessionId], references: [id])
  feedback      Feedback?
}

model Feedback {
  id        String   @id @default(uuid())
  messageId String   @unique
  rating    Int      // 1 = bad, 5 = good
  comment   String?
  createdAt DateTime @default(now())
  message   Message  @relation(fields: [messageId], references: [id])
}
```

### Project folder structure
```
rag-knowledge-chat/
├── docker-compose.yml
├── .env
├── .gitignore
├── package.json
├── tsconfig.json
├── prisma/
│   └── schema.prisma
├── src/
│   ├── index.ts              # entry point
│   ├── server.ts             # Express setup
│   ├── routes/
│   │   ├── health.ts
│   │   ├── auth.ts
│   │   ├── documents.ts
│   │   └── chat.ts
│   ├── services/
│   │   ├── llm/
│   │   │   └── openai.ts     # LLM wrapper
│   │   ├── rag/
│   │   │   ├── ingest.ts     # URL fetch + chunk
│   │   │   ├── embed.ts      # embeddings
│   │   │   └── retrieve.ts   # vector search
│   │   └── auth/
│   │       └── password.ts
│   ├── db/
│   │   └── client.ts         # Prisma client
│   └── utils/
│       └── validation.ts     # Zod schemas
├── eval/
│   ├── questions.json
│   └── run-eval.ts
└── README.md
```

---

## Summary: Your 10-Week Milestones

| Week | Focus | Milestone |
|------|-------|-----------|
| 1 | Setup + Postgres | API + DB running, users/orgs tables |
| 2 | Ingestion | URL fetch + chunking works |
| 3 | LLM basics | Can call OpenAI, get structured response |
| 4 | Embeddings | Chunks have vectors, similarity search works |
| 5 | RAG chat | `/chat` returns answer + citations |
| 6 | Polish RAG | Streaming, better prompts, refusal logic |
| 7 | Evaluation | `npm run eval` with 20+ test cases |
| 8 | Monitoring | Logs + feedback + basic dashboard |
| 9 | Security | RBAC + prompt injection defenses |
| 10 | Deploy | Live demo + README + portfolio ready |

---

## Quick Reference: Key Resources

### 🏆 Tier 1: Official Docs (Always Use First)
| Resource | URL | Why |
|----------|-----|-----|
| OpenAI Docs | https://platform.openai.com/docs | Your main LLM provider |
| Anthropic Prompt Guide | https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview | Best prompt engineering guide |
| Prisma Docs | https://www.prisma.io/docs | ORM you'll use |
| pgvector | https://github.com/pgvector/pgvector | Vector DB extension |
| LlamaIndex Docs | https://docs.llamaindex.ai/ | Production RAG patterns |

### 🏆 Tier 1: Essential Guides
| Resource | URL | Why |
|----------|-----|-----|
| Prompt Engineering Guide | https://www.promptingguide.ai/ | Most comprehensive, cited everywhere |
| OWASP LLM Top 10 | https://owasp.org/www-project-top-10-for-large-language-model-applications/ | Security standard |
| OpenAI Cookbook | https://cookbook.openai.com/ | Real-world code examples |
| MTEB Leaderboard | https://huggingface.co/spaces/mteb/leaderboard | Compare embedding models |

### 🎥 YouTube (Best Channels for GenAI)
| Channel | Focus | URL |
|---------|-------|-----|
| **Fireship** | Short explainers (< 10 min) | https://www.youtube.com/@Fireship |
| **AI Jason** | RAG tutorials (practical) | https://www.youtube.com/@AIJasonZ |
| **Sam Witteveen** | RAG deep dives (advanced) | https://www.youtube.com/@samwitteveenai |
| **Andrej Karpathy** | LLM internals (theory) | https://www.youtube.com/@AndrejKarpathy |
| **James Briggs** | Pinecone/RAG tutorials | https://www.youtube.com/@jamesbriggs |

### 💬 Communities (Ask Questions Here)
| Community | URL | Best For |
|-----------|-----|----------|
| r/LocalLLaMA | https://www.reddit.com/r/LocalLLaMA/ | Local models, Ollama |
| Hugging Face Discord | https://huggingface.co/join/discord | Models, datasets |
| LangChain Discord | https://discord.gg/langchain | RAG, chains, agents |
| OpenAI Developer Forum | https://community.openai.com/ | API questions |

### 📊 Comparison Tools (For Decisions)
| Tool | URL | What It Compares |
|------|-----|------------------|
| MTEB Leaderboard | https://huggingface.co/spaces/mteb/leaderboard | Embedding models |
| Chatbot Arena | https://lmarena.ai/ | LLM quality rankings |
| Vector DB Comparison | https://superlinked.com/vector-db-comparison | Vector databases |

---

## Next Steps

1. **Today:** Setup your project folder, install Docker, create `docker-compose.yml`
2. **This week:** Complete Week 1 milestones (API + DB + basic tables)
3. **Ask me:** When you're stuck on any step, I'll help debug or explain

Good luck. Consistency > intensity. 🚀
