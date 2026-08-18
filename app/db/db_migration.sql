-- ============================================================
--  MaaS — Complete Supabase Migration
--  Paste this entire file into Supabase SQL Editor and run.
--  Order matters — run top to bottom.
-- ============================================================


-- ============================================================
--  STEP 1: EXTENSIONS
-- ============================================================

-- Enable in Supabase Dashboard → Database → Extensions → pg_cron (do this manually)
CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector: embeddings

-- ============================================================
--  STEP 2: TABLES
-- ============================================================

-- ------------------------------------------------------------
--  tenants
--  One row per developer / company using your API.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tenants (
  tenant_id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  name               TEXT        NOT NULL,
  email              TEXT        NOT NULL UNIQUE,
  plan               TEXT        NOT NULL DEFAULT 'free'
                                 CHECK (plan IN ('free', 'pro', 'enterprise')),
  mode               TEXT        NOT NULL DEFAULT 'hosted'
                                 CHECK (mode IN ('hosted', 'byod')),
  byod_supabase_url  TEXT,
  byod_supabase_key  TEXT,
  is_suspended       BOOLEAN     NOT NULL DEFAULT FALSE,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT byod_fields_together CHECK (
    (mode = 'byod'   AND byod_supabase_url IS NOT NULL AND byod_supabase_key IS NOT NULL)
    OR
    (mode = 'hosted' AND byod_supabase_url IS NULL     AND byod_supabase_key IS NULL)
  )
);

COMMENT ON TABLE  tenants                  IS 'Developer accounts that use the MaaS API.';
COMMENT ON COLUMN tenants.plan             IS 'free | pro | enterprise — controls limits.';
COMMENT ON COLUMN tenants.mode             IS 'hosted = we store data | byod = client owns DB.';
COMMENT ON COLUMN tenants.byod_supabase_key IS 'Store AES-encrypted — never plaintext.';


-- ------------------------------------------------------------
--  api_keys
--  SHA-256 hashed keys. Raw key shown once, never stored.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS api_keys (
  id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id    UUID        NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  key_hash     TEXT        NOT NULL UNIQUE,   -- SHA-256(raw_key)
  name         TEXT,                          -- e.g. "production", "staging"
  is_active    BOOLEAN     NOT NULL DEFAULT TRUE,
  last_used_at TIMESTAMPTZ,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  api_keys          IS 'Hashed API keys — raw key shown once at creation.';
COMMENT ON COLUMN api_keys.key_hash IS 'SHA-256 hex digest of the raw rm_xxx key.';


-- ------------------------------------------------------------
--  memories
--  Core table. Every agent memory lives here.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
  id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id     UUID        NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  user_id       TEXT        NOT NULL,         -- end-user the agent is serving
  agent_id      TEXT        NOT NULL,         -- which agent stored this memory
  content       TEXT        NOT NULL CHECK (char_length(content) BETWEEN 1 AND 10000),
  embedding     VECTOR(384),                 -- BAAI/bge-small-en-v1.5 produces 384-dim embeddings
  memory_type   TEXT        NOT NULL DEFAULT 'episodic'
                            CHECK (memory_type IN ('episodic', 'semantic', 'summary')),
  metadata      JSONB       NOT NULL DEFAULT '{}',
  importance    FLOAT       NOT NULL DEFAULT 0.5
                            CHECK (importance BETWEEN 0.0 AND 1.0),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_accessed TIMESTAMPTZ,
  expires_at    TIMESTAMPTZ                   -- NULL = permanent
);

COMMENT ON TABLE  memories              IS 'Agent memories with vector embeddings.';
COMMENT ON COLUMN memories.user_id      IS 'Identifier for the end-user the agent serves.';
COMMENT ON COLUMN memories.agent_id     IS 'Identifier for the specific agent.';
COMMENT ON COLUMN memories.embedding    IS '384-dim for BAAI/bge-small-en-v1.5, 1024-dim for Voyage AI.';
COMMENT ON COLUMN memories.memory_type  IS 'episodic=events | semantic=facts | summary=compressed.';
COMMENT ON COLUMN memories.importance   IS 'Client-set priority 0.0 (low) to 1.0 (high).';
COMMENT ON COLUMN memories.expires_at   IS 'NULL means permanent. Set via ttl_days on insert.';


-- ------------------------------------------------------------
--  usage_logs
--  Every API call logged here for billing + plan enforcement.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS usage_logs (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id   UUID        NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  endpoint    TEXT        NOT NULL,           -- e.g. POST /memories, GET /memories/search
  tokens_used INT         NOT NULL DEFAULT 0, -- embedding tokens consumed
  latency_ms  INT,                            -- total response time
  status_code INT         NOT NULL,           -- 201, 200, 401, 429, 500, etc.
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE usage_logs IS 'Per-request log for billing, rate limiting, and analytics.';


-- ------------------------------------------------------------
--  memory_counts
--  Running total of stored memories per tenant.
--  Populated automatically by trg_init_memory_count trigger.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_counts (
  tenant_id   UUID    PRIMARY KEY REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  total       BIGINT  NOT NULL DEFAULT 0
);

COMMENT ON TABLE  memory_counts           IS 'Cached memory count per tenant, maintained by trigger.';
COMMENT ON COLUMN memory_counts.tenant_id IS 'FK to tenants — one row per tenant.';
COMMENT ON COLUMN memory_counts.total     IS 'Running total of stored memories for plan-limit checks.';


-- ============================================================
--  STEP 3: INDEXES
-- ============================================================

-- api_keys: fast hash lookup on every authenticated request
CREATE INDEX IF NOT EXISTS api_keys_hash_idx
  ON api_keys (key_hash);

-- api_keys: get all keys for a tenant
CREATE INDEX IF NOT EXISTS api_keys_tenant_idx
  ON api_keys (tenant_id);

-- memories: vector similarity search (ivfflat — best for <1M rows)
-- lists=100 is good up to ~1M vectors. Increase to 200 beyond that.
CREATE INDEX IF NOT EXISTS memories_vector_idx
  ON memories USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- memories: fast scoped lookup (tenant + user + agent)
CREATE INDEX IF NOT EXISTS memories_lookup_idx
  ON memories (tenant_id, user_id, agent_id);

-- memories: TTL expiry scan — partial index, only rows that expire
CREATE INDEX IF NOT EXISTS memories_ttl_idx
  ON memories (expires_at)
  WHERE expires_at IS NOT NULL;

-- memories: sort by recency within a scope
CREATE INDEX IF NOT EXISTS memories_recency_idx
  ON memories (tenant_id, user_id, agent_id, created_at DESC);

-- usage_logs: analytics + daily limit checks per tenant
CREATE INDEX IF NOT EXISTS usage_logs_tenant_date_idx
  ON usage_logs (tenant_id, created_at DESC);


-- ============================================================
--  STEP 4: ROW LEVEL SECURITY
-- ============================================================

-- Enable RLS — tenants can NEVER see each other's data
ALTER TABLE tenants    ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys   ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories   ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_logs ENABLE ROW LEVEL SECURITY;

-- Your FastAPI service uses the service_role key, so it bypasses RLS by default.
-- These policies are a safety net if you ever expose tables directly.

-- tenants: only see your own row
CREATE POLICY tenants_isolation ON tenants
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', TRUE)::UUID);

-- api_keys: only see your own keys
CREATE POLICY api_keys_isolation ON api_keys
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', TRUE)::UUID);

-- memories: core isolation — the most critical policy
CREATE POLICY memories_isolation ON memories
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', TRUE)::UUID);

-- usage_logs: only see your own usage
CREATE POLICY usage_logs_isolation ON usage_logs
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', TRUE)::UUID);


-- ============================================================
--  STEP 5: SEARCH FUNCTION (pgvector RPC)
--  Called by FastAPI as: supabase.rpc("match_memories", {...})
-- ============================================================

CREATE OR REPLACE FUNCTION match_memories(
  query_embedding  VECTOR(384),
  match_count      INT,
  p_tenant_id      UUID,
  p_user_id        TEXT,
  p_agent_id       TEXT,
  p_memory_type    TEXT DEFAULT NULL
)
RETURNS TABLE (
  id            UUID,
  content       TEXT,
  user_id       TEXT,
  agent_id      TEXT,
  memory_type   TEXT,
  metadata      JSONB,
  importance    FLOAT,
  created_at    TIMESTAMPTZ,
  last_accessed TIMESTAMPTZ,
  expires_at    TIMESTAMPTZ,
  similarity    FLOAT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    id,
    content,
    user_id,
    agent_id,
    memory_type,
    metadata,
    importance,
    created_at,
    last_accessed,
    expires_at,
    1 - (embedding <=> query_embedding) AS similarity
  FROM memories
  WHERE
    tenant_id   = p_tenant_id
    AND user_id = p_user_id
    AND agent_id = p_agent_id
    AND (p_memory_type IS NULL OR memory_type = p_memory_type)
    AND (expires_at IS NULL OR expires_at > NOW())
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;

COMMENT ON FUNCTION match_memories IS
  'Cosine similarity search over memories for a tenant+user+agent scope.
   Returns candidates for re-ranking with hybrid scoring in FastAPI.';


-- ============================================================
--  STEP 6: TTL EXPIRY CRON JOB
--  Runs every hour, deletes expired memories automatically.
--  Requires pg_cron extension (enable in Supabase dashboard first).
-- ============================================================

SELECT cron.schedule(
  'expire-memories',           -- job name
  '0 * * * *',                 -- every hour at :00
  $$
    DELETE FROM memories
    WHERE expires_at IS NOT NULL
      AND expires_at < NOW();
  $$
);


-- ============================================================
--  STEP 7: PLAN LIMITS VIEW
--  FastAPI queries this view to enforce free/pro/enterprise caps.
-- ============================================================

CREATE OR REPLACE VIEW tenant_usage AS
SELECT
  t.tenant_id                                        AS tenant_id,
  t.plan,
  t.is_suspended,

  -- Total memories stored
  COUNT(m.id)                                       AS total_memories,

  -- Requests today
  COUNT(ul.id) FILTER (
    WHERE ul.created_at >= CURRENT_DATE
  )                                                 AS requests_today,

  -- Plan limits (reference only — enforced in FastAPI)
  CASE t.plan
    WHEN 'free'       THEN 500
    WHEN 'pro'        THEN 50000
    WHEN 'enterprise' THEN 2147483647
  END                                               AS memory_limit,

  CASE t.plan
    WHEN 'free'       THEN 100
    WHEN 'pro'        THEN 10000
    WHEN 'enterprise' THEN 2147483647
  END                                               AS daily_request_limit

FROM tenants t
LEFT JOIN memories   m  ON m.tenant_id  = t.tenant_id
LEFT JOIN usage_logs ul ON ul.tenant_id = t.tenant_id
GROUP BY t.tenant_id, t.plan, t.is_suspended;

COMMENT ON VIEW tenant_usage IS
  'Live usage snapshot per tenant. Query before each request to enforce plan limits.';


-- ============================================================
--  STEP 8: TENANT INIT TRIGGER
--  Inserts a memory_counts row automatically whenever a new
--  tenant is created, so plan-limit checks always have a row.
-- ============================================================

CREATE OR REPLACE FUNCTION init_tenant_memory_count()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO memory_counts (tenant_id, total)
  VALUES (NEW.tenant_id, 0)
  ON CONFLICT (tenant_id) DO NOTHING;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION init_tenant_memory_count IS
  'Trigger function: seeds a memory_counts row for every new tenant.';

DROP TRIGGER IF EXISTS trg_init_memory_count ON tenants;

CREATE TRIGGER trg_init_memory_count
  AFTER INSERT ON tenants
  FOR EACH ROW EXECUTE FUNCTION init_tenant_memory_count();


-- ============================================================
--  STEP 9: MEMORY COUNTS AUTO-UPDATE TRIGGERS
--  Automatically increments/decrements memory_counts.total
--  when memories are inserted or deleted.
-- ============================================================

CREATE OR REPLACE FUNCTION increment_memory_count_on_insert()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO memory_counts (tenant_id, total)
  VALUES (NEW.tenant_id, 1)
  ON CONFLICT (tenant_id) DO UPDATE
  SET total = memory_counts.total + 1;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION increment_memory_count_on_insert IS
  'Trigger function: increments memory count when a memory is stored.';

DROP TRIGGER IF EXISTS trg_increment_memory_count ON memories;

CREATE TRIGGER trg_increment_memory_count
  AFTER INSERT ON memories
  FOR EACH ROW EXECUTE FUNCTION increment_memory_count_on_insert();


CREATE OR REPLACE FUNCTION decrement_memory_count_on_delete()
RETURNS TRIGGER AS $$
BEGIN
  UPDATE memory_counts
  SET total = GREATEST(0, total - 1)
  WHERE tenant_id = OLD.tenant_id;
  RETURN OLD;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION decrement_memory_count_on_delete IS
  'Trigger function: decrements memory count when a memory is deleted/expired.';

DROP TRIGGER IF EXISTS trg_decrement_memory_count ON memories;

CREATE TRIGGER trg_decrement_memory_count
  AFTER DELETE ON memories
  FOR EACH ROW EXECUTE FUNCTION decrement_memory_count_on_delete();


-- ============================================================
--  DONE
-- ============================================================
-- Tables:     tenants, api_keys, memories, usage_logs, memory_counts
-- Indexes:    7 indexes covering auth, vector search, TTL, recency
-- Security:   RLS on all 4 tables
-- Functions:  match_memories (vector search RPC)
--             init_tenant_memory_count (trigger function)
--             increment_memory_count_on_insert (trigger function)
--             decrement_memory_count_on_delete (trigger function)
-- Triggers:   trg_init_memory_count (seeds memory_counts on tenant insert)
--             trg_increment_memory_count (increments memory_counts on insert)
--             trg_decrement_memory_count (decrements memory_counts on delete)
-- Cron:       expire-memories (hourly TTL cleanup)
-- Views:      tenant_usage (plan enforcement)
-- ============================================================