from app.db.client import get_master_client
from dotenv import load_dotenv
import asyncio
load_dotenv()

from app.schemas.memory import (
    GetContext, MemoryCreate, MemoryDelete,
    MemorySearch, MemoryWipe, MemoryUpdate,
)
from app.services.memory import (
    get_context, is_duplicate, store_memory, search_memories,
    delete_memory, update_memory, wipe_memories, expire_memories,
)
from app.schemas.memory import IsDuplicate

TENANT_ID = "cde6cfd6-4224-48da-8e5f-abcf25b9a01a"
USER_ID   = "user_001"
AGENT_ID  = "support_bot"
db        = get_master_client()


# ── Payloads ──────────────────────────────────────────────────────

load_1 = MemoryCreate(
    user_id=USER_ID, agent_id=AGENT_ID,
    content="User is based in Lagos, nigeria",
    memory_type="semantic", importance=0.8,
)
load_2 = MemoryCreate(
    user_id=USER_ID, agent_id=AGENT_ID,
    content="User prefers concise bullet points",
    memory_type="semantic", importance=0.7,
)
load_3 = MemoryCreate(
    user_id=USER_ID, agent_id=AGENT_ID,
    content="User is building a customer support agent",
    memory_type="semantic", importance=0.6,
)
load_4 = MemoryCreate(
    user_id=USER_ID, agent_id=AGENT_ID,
    content="Temporary note",
    memory_type="semantic", importance=0.5,
    ttl_days=1,
)

context_payload       = GetContext(user_id=USER_ID,      agent_id=AGENT_ID)
ghost_context_payload = GetContext(user_id="ghost_user", agent_id=AGENT_ID)
dup_check_payload = IsDuplicate(user_id=USER_ID, agent_id=AGENT_ID)

search_load_1 = MemorySearch(
    query="Where does this user live?",
    user_id=USER_ID, agent_id=AGENT_ID,
    top_k=3, min_score=0.0,
)


# ── 0. CLEAN SLATE ────────────────────────────────────────────────
# Guarantees no leftover data from previous runs interferes

print("=== 0. CLEAN SLATE ===")
asyncio.run(wipe_memories(
    payload=MemoryWipe(user_id=USER_ID, agent_id=AGENT_ID, confirm=True),
    db=db, tenant_id=TENANT_ID,
))
print("  🧹 Previous test data wiped\n")


# ── 1. STORE ──────────────────────────────────────────────────────

print("=== 1. STORE ===")

mem_1 = asyncio.run(store_memory(payload=load_1, db=db, tenant_id=TENANT_ID))
mem_2 = asyncio.run(store_memory(payload=load_2, db=db, tenant_id=TENANT_ID))
mem_3 = asyncio.run(store_memory(payload=load_3, db=db, tenant_id=TENANT_ID))
mem_4 = asyncio.run(store_memory(payload=load_4, db=db, tenant_id=TENANT_ID))

assert mem_1 is not None, "store_memory failed for mem_1"
assert mem_2 is not None, "store_memory failed for mem_2"
assert mem_3 is not None, "store_memory failed for mem_3"
assert mem_4 is not None, "store_memory failed for mem_4"

print(f"  mem_1: {mem_1.id}")
print(f"  mem_2: {mem_2.id}")
print(f"  mem_3: {mem_3.id}")
print(f"  mem_4: {mem_4.id}")


# ── 2. SEARCH ─────────────────────────────────────────────────────

print("\n=== 2. SEARCH ===")

results = asyncio.run(search_memories(
    payload=search_load_1, db=db, tenant_id=TENANT_ID,
))
assert len(results.memories) > 0, "Expected search results"
for r in results.memories:
    print(f"  [{r.score:.3f}] {r.content}")


# ── 3. GET CONTEXT ────────────────────────────────────────────────

print("\n=== 3. GET CONTEXT ===")

context = asyncio.run(get_context(
    payload=context_payload, tenant_id=TENANT_ID, db=db, top_k=10,
))

print(f"  Total memories in context: {context.total}")
print("  Ranked by importance + recency:")
for m in context.memories:
    print(f"    [importance={m.importance}] {m.content}")

assert context.total >= 3, f"Expected at least 3, got {context.total}"
assert context.memories[0].importance >= context.memories[-1].importance, \
    "Memories should be sorted by importance descending"


# ── 4. IS DUPLICATE ───────────────────────────────────────────────

print("\n=== 4. IS DUPLICATE ===")

# Should be duplicate — similar to load_1
dup_check = asyncio.run(is_duplicate(
    content="User is based in Lagos, nigeria",
    payload=dup_check_payload,
    tenant_id=TENANT_ID,
    db=db,
    threshold=0.95,
))
print(f"  'User lives in Lagos, Nigeria' is duplicate: {dup_check}")
assert dup_check is True, "Expected duplicate to be detected"

# Should NOT be duplicate — completely different content
unique_check = asyncio.run(is_duplicate(
    content="User has a dog named Max",
    payload=IsDuplicate(user_id=USER_ID, agent_id=AGENT_ID),
    tenant_id=TENANT_ID,
    db=db,
    threshold=0.95,
))
print(f"  'User has a dog named Max' is duplicate: {unique_check}")
assert unique_check is False, "Expected unique content to pass"


# ── 5. TTL EXPIRY ─────────────────────────────────────────────────

print("\n=== 5. TTL EXPIRY ===")

expired = asyncio.run(expire_memories(db=db))
print(f"  Expired: {expired}")
# mem_4 has ttl_days=1 so it won't expire immediately
# this confirms the function runs without error


# ── 6. DELETE ONE ─────────────────────────────────────────────────

print("\n=== 6. DELETE ONE ===")

assert mem_1 is not None and mem_1.id, "mem_1 must exist for deletion test"

del_mem = asyncio.run(delete_memory(
    payload=MemoryDelete(
        user_id=USER_ID,
        agent_id=AGENT_ID,
        memory_id=mem_1.id,
    ),
    db=db,
    tenant_id=TENANT_ID,
))
print(f"  Deleted: {del_mem.deleted}")
assert del_mem.deleted == 1, "Expected 1 memory deleted"


# ── 7. SEARCH AGAIN (Lagos should be gone) ────────────────────────

print("\n=== 7. SEARCH AGAIN (Lagos should be gone) ===")

results = asyncio.run(search_memories(
    payload=search_load_1, db=db, tenant_id=TENANT_ID,
))
contents = [r.content for r in results.memories]
print(f"  Results: {len(results.memories)}")
for r in results.memories:
    print(f"  [{r.score:.3f}] {r.content}")
assert "User is based in Lagos, nigeria" not in contents, \
    "Lagos memory should be gone after delete"


# ── 8. UPDATE MEMORY ──────────────────────────────────────────────

print("\n=== 8. UPDATE MEMORY ===")

# mem_2 still exists — use it for update
assert mem_2 is not None and mem_2.id, "mem_2 must exist for update test"

updated_mem = asyncio.run(update_memory(
    payload=MemoryUpdate(
        user_id=USER_ID,
        agent_id=AGENT_ID,
        memory_id=mem_2.id,
        new_content="User moved from Lagos to Abuja, Nigeria",
        importance=0.95,
    ),
    tenant_id=TENANT_ID,
    db=db,
))

print(f"  Memory ID unchanged: {updated_mem.id == mem_2.id}")
print(f"  New content: {updated_mem.content}")
print(f"  New importance: {updated_mem.importance}")

assert updated_mem.id == mem_2.id, "Memory ID should not change on update"
assert updated_mem.content == "User moved from Lagos to Abuja, Nigeria"
assert updated_mem.importance == 0.95

# Verify update reflected in context
print("\n  Verifying update in get_context()...")
context_after = asyncio.run(get_context(
    payload=context_payload, tenant_id=TENANT_ID, db=db, top_k=10,
))
contents = [m.content for m in context_after.memories]
assert "User moved from Lagos to Abuja, Nigeria" in contents, \
    "Updated content should appear in context"
assert "User prefers concise bullet points" not in contents, \
    "Old content should be gone"
print("  ✅ Context reflects updated memory correctly")


# ── 9. UPDATE WITH INVALID ID (should raise) ──────────────────────

print("\n=== 9. UPDATE WITH INVALID ID (should raise) ===")

try:
    asyncio.run(update_memory(
        payload=MemoryUpdate(
            user_id=USER_ID,
            agent_id=AGENT_ID,
            memory_id="00000000-0000-0000-0000-000000000000",
            new_content="This should fail",
            importance=0.5,
        ),
        tenant_id=TENANT_ID,
        db=db,
    ))
    print("  ❌ Should have raised ValueError")
except (ValueError, RuntimeError) as e:
    print(f"  ✅ Caught expected error: {e}")


# ── 10. GET CONTEXT FOR UNKNOWN USER ──────────────────────────────

print("\n=== 10. GET CONTEXT FOR UNKNOWN USER (should return empty) ===")

empty_context = asyncio.run(get_context(
    payload=ghost_context_payload,
    tenant_id=TENANT_ID,
    db=db,
    top_k=10,
))
print(f"  Memories returned: {empty_context.total}")
assert empty_context.total == 0, "Unknown user should have no memories"
print("  ✅ Correctly returned empty context")


# ── 11. WIPE ALL ──────────────────────────────────────────────────

print("\n=== 11. WIPE ALL ===")

wipe_result = asyncio.run(wipe_memories(
    payload=MemoryWipe(user_id=USER_ID, agent_id=AGENT_ID, confirm=True),
    db=db, tenant_id=TENANT_ID,
))
print(f"  Wiped: {wipe_result.deleted}")
assert wipe_result.deleted > 0, "Expected at least 1 memory wiped"


# ── 12. SEARCH AFTER WIPE (should return nothing) ─────────────────

print("\n=== 12. SEARCH AFTER WIPE (should return nothing) ===")

results = asyncio.run(search_memories(
    payload=MemorySearch(
        query="anything",
        user_id=USER_ID,
        agent_id=AGENT_ID,
        top_k=3,
        min_score=0.0,
    ),
    db=db,
    tenant_id=TENANT_ID,
))
print(f"  Results: {len(results.memories)}")
assert len(results.memories) == 0, "Expected 0 results after wipe"

print("\n✅ All tests passed.")

test_userApi = "maas_E_lhk7BHJ8HpLth7h2MxX2UW0pepJeYnivlU6KRsC0o"