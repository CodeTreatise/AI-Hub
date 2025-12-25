# GenAI/RAG Developer Course - Design Document

## Target Learner Profile
- **Background:** Node.js + Angular developer
- **Level:** Self-described "below average" - needs things explained clearly
- **Goal:** Become Applied GenAI Engineer / RAG Developer
- **Learning Style:** Needs to understand WHY, not just HOW

---

## Course Philosophy

### 1. No Assumptions
- Every concept is explained from scratch
- Jargon is defined when first used
- "Obvious" things are stated explicitly

### 2. Always Answer These Questions
For every topic:
- **WHAT** is it? (Definition in simple terms)
- **WHY** do we need it? (The problem it solves)
- **HOW** does it work? (Mechanics, step by step)
- **WHEN** do we use it? (Real scenarios)
- **HOW** does it connect? (Links to other concepts)

### 3. Living Content
- All content is fetched from real sources
- Sources are credited and linked
- Can be refreshed/updated
- Not static training data

### 4. Test Understanding
- Each module has self-check questions
- Exercises to apply knowledge
- "Explain it back" prompts

---

## Module Structure

Each module contains:

```
module-X-name/
├── 00-overview.html       # What you'll learn, prerequisites, time estimate
├── 01-concept.html        # Theory explained simply
├── 02-deep-dive.html      # Detailed mechanics
├── 03-hands-on.html       # Code you run, step by step
├── 04-gotchas.html        # Common mistakes, debugging tips
├── 05-test.html           # Quiz, exercises
├── 06-connections.html    # How this links to other modules
└── references.json        # Source URLs, further reading
```

---

## Content Format (HTML Template)

```html
<article class="module-content">
  <header>
    <h1>Module Title</h1>
    <div class="meta">
      <span class="time">⏱️ ~30 min</span>
      <span class="difficulty">📊 Beginner</span>
    </div>
  </header>

  <section class="learning-objectives">
    <h2>🎯 What You'll Learn</h2>
    <ul>
      <li>Objective 1</li>
      <li>Objective 2</li>
    </ul>
  </section>

  <section class="prerequisites">
    <h2>📋 Before You Start</h2>
    <p>Make sure you understand: [links to prior modules]</p>
  </section>

  <section class="content">
    <h2>📖 The Concept</h2>
    <!-- Main teaching content -->
  </section>

  <section class="why-it-matters">
    <h2>💡 Why This Matters</h2>
    <p>How this connects to your goal...</p>
  </section>

  <section class="hands-on">
    <h2>🛠️ Try It Yourself</h2>
    <!-- Code blocks, exercises -->
  </section>

  <section class="common-mistakes">
    <h2>⚠️ Common Mistakes</h2>
    <!-- Gotchas, tips -->
  </section>

  <section class="test-yourself">
    <h2>✅ Check Your Understanding</h2>
    <!-- Quiz questions -->
  </section>

  <section class="connections">
    <h2>🔗 How This Connects</h2>
    <!-- Links to related modules -->
  </section>

  <footer class="references">
    <h2>📚 Sources & Further Reading</h2>
    <ul>
      <li><a href="...">Source 1</a></li>
    </ul>
  </footer>
</article>
```

---

## Module Outline

### Module 0: The Big Picture
**Why are you here?**
- What is GenAI / LLMs / RAG
- Why this is valuable (career, problems solved)
- What you'll build in this course
- The tech stack and why each piece

### Module 1: Your Dev Environment
**Set up properly before coding**
- Terminal basics (for those who need it)
- Docker explained (not just commands)
  - What are containers vs VMs
  - Images vs Containers
  - Docker Compose for multi-service apps
  - Debugging containers
- VS Code setup for this course
- Git basics (if needed)

### Module 2: Database Foundations
**Data is everything in AI**
- Why databases matter for AI
- SQL fundamentals (SELECT, INSERT, JOIN)
- Postgres specifically (why not MySQL)
- Connecting from Node.js (Prisma)
- What is pgvector (preview)

### Module 3: Node.js + TypeScript
**Your server foundation**
- Project structure best practices
- TypeScript for safety
- Express.js basics
- REST API design
- Environment variables, config
- Error handling patterns

### Module 4: LLM Fundamentals
**Understanding the AI part**
- What is a Large Language Model
- Tokens, context windows, limits
- Temperature, top_p (what they mean)
- System vs User prompts
- Calling the OpenAI API
- Streaming responses
- Cost awareness

### Module 5: Embeddings & Vectors
**The math made simple**
- What are embeddings (intuition)
- Why vectors represent meaning
- Similarity search explained
- pgvector in practice
- Choosing embedding models

### Module 6: RAG Architecture
**Putting it all together**
- The RAG pipeline visualized
- Document ingestion
- Chunking strategies
- Retrieval strategies
- Generation with context
- Evaluation basics

### Module 7: Building Your RAG App
**Hands-on project**
- Step-by-step build
- Each file explained
- Testing your app
- Debugging tips

### Module 8: Production Concerns
**Making it real**
- Security (prompt injection, data leaks)
- Performance (caching, batching)
- Monitoring & logging
- Deployment basics

### Module 9: Frontend Integration
**Using your Angular skills**
- Calling your RAG API
- Streaming in the browser
- Chat UI patterns
- Error handling for AI

---

## Source Strategy

For each topic, we will:
1. Identify 3-5 authoritative sources
2. Fetch and extract key content
3. Synthesize into our format
4. Credit all sources
5. Provide links for deeper reading

### Priority Sources:
- Official documentation (Docker, Postgres, OpenAI, Prisma)
- GitHub READMEs (pgvector, LangChain.js)
- Reputable tutorials (Pinecone, Supabase guides)
- Video transcripts (for visual learners)

---

## Next Steps

1. ✅ Design complete (this document)
2. 🔄 Build content fetcher
3. 🔄 Create HTML templates
4. 🔄 Start with Module 0
5. 🔄 Iterate based on content quality
