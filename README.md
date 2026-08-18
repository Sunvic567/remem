# Remem — Memory-as-a-Service for AI Agents

> Persistent, semantically-ranked memory for any AI agent. Two API calls. No infrastructure to manage.

[![PyPI version](https://badge.fury.io/py/remem-py.svg)](https://pypi.org/project/remem-py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Every AI agent forgets everything the moment a session ends. Developers are left to either:

- Dump full conversation history into every prompt (expensive, hits token limits)
- Build custom vector store logic per project (weeks of work)
- Accept stateless agents that frustrate users

## The Solution

```python
pip install remem-py
```

```python
from remem import RememClient

client = RememClient(api_key="your-key")

# Store a memory
client.store("User is on the Pro plan and prefers email contact")

# Retrieve relevant context before generating a response
context = client.get_context("What do I know about this user?")
```

That is the entire integration. Your agent now has persistent memory.

---

## What Makes Remem Different

### Hybrid Scoring — Not Dumb Retrieval
Most memory solutions do cosine similarity and call it done. Remem scores every
memory on three axes simultaneously:

| Factor | Weight | What it does |
|--------|--------|-------------|
| Semantic relevance | 70% | Cosine similarity to query |
| Recency decay | 20% | Recent memories rank higher |
| Importance weighting | 10% | Developer-assigned priority |

The agent gets the *right* memories, not just the most similar ones.

### Memory That Manages Itself
- **TTL expiry** via pg_cron — memories age out automatically
- **Duplicate detection** at 0.95 cosine similarity — no redundant storage
- **Auto-pruning** — the memory layer runs itself

### BYOD — Data Sovereignty for Enterprise
Remem's Bring Your Own Database option lets enterprise customers connect their
own Supabase instance. Remem runs the logic. The data never touches Remem's servers.

---

## AMD Integration

This branch integrates AMD hardware for embedding generation via Fireworks AI,
running on AMD Instinct MI300X GPUs through ROCm.

```
User Request
     ↓
Remem API (FastAPI)
     ↓
Fireworks AI Embedding Service
(AMD Instinct MI300X via ROCm)
     ↓
nomic-embed-text-v1.5 @ 768 dimensions
     ↓
Supabase pgvector
Hybrid Scoring Engine
     ↓
Ranked Memory Results
     ↓
Agent Context Window
```

**Benchmark endpoint:** `GET /benchmark`

```json
{
  "status": "ok",
  "amd_integration": {
    "provider": "Fireworks AI",
    "hardware": "AMD Instinct MI300X",
    "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
    "dimensions": 768
  },
  "embedding_benchmark": {
    "avg_ms": 187.45,
    "min_ms": 142.30,
    "max_ms": 241.80,
    "samples": 5,
    "errors": 0
  }
}
```

| Metric | Result |
|--------|--------|
| Embedding avg latency | 418.73ms |
| Embedding min latency | 312.89ms |
| Embedding max latency | 507.28ms |
| Samples | 5 |
| Errors | 0 |
| Embedding model | nomic-embed-text-v1.5 |
| Vector dimensions | 768 |
| Hardware | AMD Instinct MI300X via Fireworks AI |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- Supabase account
- Fireworks AI API key

### Run Locally

```bash
git clone https://github.com/sunvic567/remem
cd remem
git checkout amd-hackathon

cp .env.example .env
# Fill in your credentials

docker compose up
```

API: `http://localhost:8000`
Docs: `http://localhost:8000/docs`
Benchmark: `http://localhost:8000/benchmark`

### Environment Variables

```bash
# Supabase
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key

# AMD / Fireworks AI
FIREWORKS_API_KEY=your-fireworks-api-key

# App
SECRET_KEY=your-secret-key
APP_ENV=development
```

---

## API Reference

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/memories` | POST | API key | Store a memory |
| `/memories/search` | GET | API key | Semantic search — hybrid scored results |
| `/memories/context` | GET | API key | Get ranked context for a user+agent pair |
| `/memories` | GET | API key | List all memories (paginated) |
| `/memories/{id}` | PATCH | API key | Update an existing memory's content/importance |
| `/memories/duplicate-check` | POST | API key | Check if a memory already exists |
| `/memories/{id}` | DELETE | API key | Delete a single memory |
| `/memories` | DELETE | API key | Wipe all memories for a user+agent (`confirm=true` required) |
| `/benchmark` | GET | — | AMD Instinct MI300X embedding latency metrics |
| `/health` | GET | — | System health check |
| `/webhooks/signup/free` | POST | — | Free plan signup (creates tenant, emails key) |
| `/webhooks/flutterwave` | POST | Signature | Flutterwave payment webhook |
| `/admin/tenants` | POST | Master key | Create a tenant (internal use) |

Full docs: https://dev.remem.online/docs

---

## Demo Agent

See `demo/support_agent.py` for a customer support agent demonstrating
persistent memory across sessions using Remem + LangChain + Google Gemini.

```bash
cd demo
pip install -r requirements.txt
cp .env.example .env
python support_agent.py
```

**What the demo shows:**

- Session 1–3 without Remem: agent asks the user to repeat themselves every time
- Session 1–3 with Remem: agent remembers name, plan, and issue from session 1

---

## Project Structure

```
remem/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── memories.py      # Core memory endpoints
│   │       ├── benchmark.py     # AMD benchmark endpoint
│   │       ├── health.py
│   │       └── webhooks.py
│   ├── core/
│   │   └── config.py
│   └── db/
├── demo/
│   ├── support_agent.py         # Demo agent
│   ├── requirements.txt
│   └── .env.example
├── docker-compose.yml
├── Dockerfile
├── SUBMISSION.md
└── README.md
```

---

## Roadmap

- [ ] Multi-modal memory (images, audio, documents)
- [ ] Memory analytics dashboard
- [ ] MCP server integration
- [ ] Memory compression and summarization
- [ ] SDKs for JavaScript and Go

---

## Links

- **Product:** https://dev.remem.online
- **Docs:** https://docs.remem.online
- **PyPI:** https://pypi.org/project/remem-py
- **Benchmark:** https://api.remem.online/benchmark

---

*Built for AMD Developer Hackathon: ACT II — Unicorn Track*