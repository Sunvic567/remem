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

client = RememClient(api_key="rm_xxxxxxxxxxxxxx")

# Store a memory
result = client.remember(
    "User prefers dark mode",
    user_id="user_123",
    agent_id="support_bot",
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

async with AsyncRememClient(api_key="rm_xxxxxxxxxxxxxx") as client:
    await client.remember(
        "User prefers concise responses",
        user_id="user_123",
        agent_id="support_bot",
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

---

## Configuration

```python
client = RememClient(
    api_key="rm_xxxxxxxxxxxxxx",   # required
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

client = RememClient(api_key="rm_xxxxxxxxxxxxxx")

try:
    result = client.remember(
        "User prefers dark mode",
        user_id="user_123",
        agent_id="support_bot",
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
with RememClient(api_key="rm_xxxxxxxxxxxxxx") as client:
    client.remember("User is in Lagos", user_id="u1", agent_id="bot")

# Async
async with AsyncRememClient(api_key="rm_xxxxxxxxxxxxxx") as client:
    await client.remember("User is in Lagos", user_id="u1", agent_id="bot")
```

---

## LangGraph Integration

```python
from langgraph.graph import StateGraph, MessagesState
from remem import AsyncRememClient

remem = AsyncRememClient(api_key="rm_xxxxxxxxxxxxxx")


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
  -H "X-API-Key: rm_xxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "user_id": "u1", "agent_id": "bot"}'

# Search
curl "https://api.remem.online/memories/search?query=preferences&user_id=u1&agent_id=bot" \
  -H "X-API-Key: rm_xxxxxxxxxxxxxx"

# Context
curl "https://api.remem.online/memories/context?user_id=u1&agent_id=bot" \
  -H "X-API-Key: rm_xxxxxxxxxxxxxx"
```

Full API reference at [remem.online/docs](https://remem.online/docs).

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
- **Version**: `0.1.4`
- **Import**: `from remem import RememClient`

---

## License

MIT — see [LICENSE](LICENSE) for details.