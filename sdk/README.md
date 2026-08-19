# remem-py

A lightweight Python SDK for [Remem](https://remem.online) — persistent memory storage and retrieval for AI agents.

[![PyPI version](https://img.shields.io/pypi/v/Remem-py.svg)](https://pypi.org/project/Remem-py/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/Remem-py.svg)](https://pypi.org/project/Remem-py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Install

```bash
pip install Remem-py
```

---

## Quick Start

```python
from remem import RememClient

client = RememClient(api_key="rm_xxxxx")

# Store a memory
result = client.remember(
    "User prefers dark mode",
    user_id="user_123",
    agent_id="support_bot",
    metadata={"theme_preference": "dark"},
)
print(result.id)

# Recall relevant memories
memories = client.recall(
    "What does this user prefer?",
    user_id="user_123",
    agent_id="support_bot",
)

for memory in memories:
    print(memory.content, memory.score)
```

---

## Async Support

Use `AsyncRememClient` for async apps and frameworks like FastAPI or LangGraph:

```python
from remem import AsyncRememClient

async with AsyncRememClient(api_key="rm_xxxxx") as client:
    await client.remember(
        "User prefers concise responses",
        user_id="user_123",
        agent_id="support_bot",
        metadata={"response_style": "concise"},
    )

    memories = await client.recall(
        "What are this user's preferences?",
        user_id="user_123",
        agent_id="support_bot",
    )
```

---

## Client Methods

| Method | Description |
|---|---|
| `remember(...)` | Store a memory for a user+agent pair |
| `recall(...)` | Semantic search — returns top-k memories by hybrid score |
| `context(...)` | Load contextual memories at session start |
| `list(...)` | List all stored memories with pagination |
| `update(...)` | Update an existing memory with new content |
| `forget(...)` | Delete a single memory by ID |
| `forget_all(...)` | Wipe all memories for a user+agent pair |
| `is_duplicate(...)` | Check whether a memory already exists before storing |

### `list()` — Paginate stored memories
```python
memories_page = client.list(
    user_id="user_123",
    agent_id="support_bot",
    limit=20,
    offset=0,
)
```

### `update()` — Update an existing memory
```python
client.update(
    memory_id="123e4567-e89b-12d3-a456-426614174000",
    user_id="user_123",
    agent_id="support_bot",
    new_content="User moved from Lagos to Abuja",
    importance=0.9,
    metadata={"verified": True},
)
```

### `forget()` and `forget_all()` — Deletion
```python
# Delete a single memory
client.forget(memory_id="...", user_id="user_123", agent_id="support_bot")

# Wipe all memories for a user+agent
client.forget_all(user_id="user_123", agent_id="support_bot")
```

### `is_duplicate()` — Check for duplicates
```python
is_dup = client.is_duplicate(
    content="User prefers dark mode",
    user_id="user_123",
    agent_id="support_bot",
    threshold=0.95,
)
```

---

## Configuration

```python
client = RememClient(
    api_key="rm_xxxxx",   # required
    base_url="https://api.remem.online",  # default
    timeout=30.0,               # seconds
)
```

---

## Error Handling

```python
from remem import (
    RememClient,
    AuthenticationError,
    PlanLimitError,
    MemoryNotFoundError,
    DuplicateMemoryError,
    RememError,
)

client = RememClient(api_key="rm_xxxxx")

try:
    result = client.remember(
        "User prefers dark mode",
        user_id="user_123",
        agent_id="support_bot",
        metadata={"theme": "dark"},
    )

except AuthenticationError:
    print("Invalid API key")

except PlanLimitError:
    print("Memory limit reached — upgrade your plan at remem.online")

except DuplicateMemoryError:
    print("This memory already exists — skipped")

except MemoryNotFoundError:
    print("Memory ID not found")

except RememError as e:
    print(f"API error {e.status_code}: {e.detail}")
```

---

## Context Manager

```python
# Sync
with RememClient(api_key="rm_xxxxx") as client:
    client.remember("User is in Lagos", user_id="u1", agent_id="bot")

# Async
async with AsyncRememClient(api_key="rm_xxxxx") as client:
    await client.remember("User is in Lagos", user_id="u1", agent_id="bot")
```

---

## LangGraph Integration

```python
from langgraph.graph import StateGraph, MessagesState
from remem import AsyncRememClient

remem = AsyncRememClient(api_key="rm_xxxxx")


async def load_memory(state: MessagesState):
    """Load memories before the agent responds."""
    last_message = state["messages"][-1].content

    result = await remem.recall(
        query=last_message,
        user_id=state["user_id"],
        agent_id="my_agent",
    )

    context = "\n".join(m.content for m in result)
    # inject context into your prompt here
    return state


async def save_memory(state: MessagesState):
    """Save what the agent learned after responding."""
    last_message = state["messages"][-1].content

    await remem.remember(
        content=last_message,
        user_id=state["user_id"],
        agent_id="my_agent",
        memory_type="episodic",
    )
    return state


# Wire into your graph
builder = StateGraph(MessagesState)
builder.add_node("load_memory", load_memory)
builder.add_node("agent", your_agent_node)
builder.add_node("save_memory", save_memory)

builder.add_edge("load_memory", "agent")
builder.add_edge("agent", "save_memory")
```

---

## Pricing

| Plan | Memories | Requests/day | Price |
|---|---|---|---|
| Free | 500 | 100 | $0/mo |
| Pro | 50,000 | 10,000 | $19/mo |
| Enterprise | Unlimited | Unlimited | $99+/mo |

Enterprise includes BYOD (Bring Your Own Database) — your data never leaves your Supabase instance.

---

## REST API

Every method maps to a REST endpoint:

```bash
# Store
curl -X POST https://api.remem.online/memories \
  -H "X-API-Key: rm_xxxxx" \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "user_id": "u1", "agent_id": "bot"}'

# Search
curl "https://api.remem.online/memories/search?query=preferences&user_id=u1&agent_id=bot" \
  -H "X-API-Key: rm_xxxxx"

# Context
curl "https://api.remem.online/memories/context?user_id=u1&agent_id=bot" \
  -H "X-API-Key: rm_xxxxx"

# List (paginate)
curl "https://api.remem.online/memories?user_id=u1&agent_id=bot&limit=20&offset=0" \
  -H "X-API-Key: rm_xxxxx"

# Update
curl -X PATCH https://api.remem.online/memories/{memory_id} \
  -H "X-API-Key: rm_xxxxx" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u1", "agent_id": "bot", "new_content": "User moved to Abuja"}'

# Delete one
curl -X DELETE "https://api.remem.online/memories/{memory_id}?user_id=u1&agent_id=bot" \
  -H "X-API-Key: rm_xxxxx"

# Wipe all
curl -X DELETE "https://api.remem.online/memories?user_id=u1&agent_id=bot&confirm=true" \
  -H "X-API-Key: rm_xxxxx"

# Duplicate check
curl "https://api.remem.online/memories/duplicate-check?content=User+prefers+dark+mode&user_id=u1&agent_id=bot" \
  -H "X-API-Key: rm_xxxxx"
```

Full API reference at [docs.remem.online](https://docs.remem.online).

---

## Links

- [Website](https://dev.remem.online)
- [API Docs](https://docs.remem.online)
- [GitHub](https://github.com/sunvic567/remem)
- [Report a Bug](https://github.com/sunvic567/remem/issues)
- [Email](mailto:support@remem.online)

---

## Package Info

- **Package**: `Remem-py`
- **Version**: `0.1.5`
- **Import**: `from remem import RememClient`

---

## License

MIT — see [LICENSE](LICENSE) for details.
