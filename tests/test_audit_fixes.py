"""
tests/test_audit_fixes.py
=========================
Automated tests covering the three areas addressed in the technical audit:
  1. API key format enforcement (rm_ prefix)
  2. Benchmark endpoint correctness
  3. Plan limit enforcement via memory_counts table
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from fastapi.testclient import TestClient


# ── Fixtures ───────────────────────────────────────────────────────

TENANT_ID = "tenant_test_audit"
USER_ID   = "user_audit"
AGENT_ID  = "audit_bot"

MOCK_EMBEDDING = [0.05] * 384


# ─────────────────────────────────────────────────────────────────
# Section 1: API Key Prefix Enforcement
# ─────────────────────────────────────────────────────────────────

class TestApiKeyFormat:
    """
    Verify that the auth layer enforces the rm_ prefix and that
    any other prefix is immediately rejected with 401.
    """

    def _get_app(self):
        from app.main import app
        return TestClient(app)

    def test_missing_api_key_returns_403(self):
        """No X-API-Key header should be rejected."""
        client = self._get_app()
        resp = client.get("/memories", params={
            "user_id": USER_ID, "agent_id": AGENT_ID
        })
        assert resp.status_code in (401, 403, 422)

    def test_wrong_prefix_returns_401(self):
        """Keys not starting with rm_ must be rejected before DB lookup."""
        client = self._get_app()
        for bad_key in ("remem_live_abc", "ml_live_xyz", "maas_abc", "sk-openai123", "Bearer token"):
            resp = client.get("/memories", headers={"X-API-Key": bad_key}, params={
                "user_id": USER_ID, "agent_id": AGENT_ID
            })
            assert resp.status_code == 401, (
                f"Key '{bad_key}' should have been rejected with 401, got {resp.status_code}"
            )

    def test_rm_prefix_passes_format_check(self):
        """
        A properly prefixed key passes the format check and only fails because
        the key doesn't exist in the DB (not because of a format error).
        The detail message should NOT mention 'format'.
        """
        client = self._get_app()
        resp = client.get("/memories", headers={"X-API-Key": "rm_fakekeythatdoesnotexist12345"}, params={
            "user_id": USER_ID, "agent_id": AGENT_ID
        })
        assert resp.status_code == 401
        body = resp.json()
        # Format error says "Keys must start with 'rm_'" — we should NOT see that message here
        assert "format" not in body.get("detail", "").lower(), (
            "rm_-prefixed key should fail due to DB lookup, not format validation"
        )


# ─────────────────────────────────────────────────────────────────
# Section 2: Benchmark Endpoint
# ─────────────────────────────────────────────────────────────────

class TestBenchmarkEndpoint:
    """
    Verify the /benchmark endpoint exists, returns valid structure,
    and correctly reports metrics.
    """

    def _get_app(self):
        from app.main import app
        return TestClient(app)

    def test_benchmark_endpoint_exists(self):
        """GET /benchmark must respond — not 404."""
        with patch("app.api.routes.benchmark.embed_for_search", return_value=MOCK_EMBEDDING):
            client = self._get_app()
            resp = client.get("/benchmark")
        assert resp.status_code != 404, "/benchmark endpoint is missing — was it registered?"

    def test_benchmark_returns_200(self):
        """GET /benchmark should return HTTP 200 OK."""
        with patch("app.api.routes.benchmark.embed_for_search", return_value=MOCK_EMBEDDING):
            client = self._get_app()
            resp = client.get("/benchmark")
        assert resp.status_code == 200

    def test_benchmark_response_has_required_fields(self):
        """Response JSON must contain all expected latency fields."""
        required_fields = {
            "avg_latency", "min_latency", "max_latency",
            "samples", "errors", "model", "dimensions", "hardware"
        }
        with patch("app.api.routes.benchmark.embed_for_search", return_value=MOCK_EMBEDDING):
            client = self._get_app()
            resp = client.get("/benchmark")
        data = resp.json()
        missing = required_fields - set(data.keys())
        assert not missing, f"Benchmark response is missing fields: {missing}"

    def test_benchmark_samples_count(self):
        """samples field must equal the expected number of probes (5)."""
        with patch("app.api.routes.benchmark.embed_for_search", return_value=MOCK_EMBEDDING):
            client = self._get_app()
            resp = client.get("/benchmark")
        assert resp.json()["samples"] == 5

    def test_benchmark_errors_zero_on_success(self):
        """When embedding succeeds, errors should be 0."""
        with patch("app.api.routes.benchmark.embed_for_search", return_value=MOCK_EMBEDDING):
            client = self._get_app()
            resp = client.get("/benchmark")
        assert resp.json()["errors"] == 0

    def test_benchmark_counts_errors_on_failure(self):
        """When embedding raises, errors should be incremented, not crash the endpoint."""
        with patch("app.api.routes.benchmark.embed_for_search", side_effect=RuntimeError("GPU unavailable")):
            client = self._get_app()
            resp = client.get("/benchmark")
        assert resp.status_code == 200
        data = resp.json()
        assert data["errors"] == 5
        assert data["avg_latency"] == "N/A"


# ─────────────────────────────────────────────────────────────────
# Section 3: Plan Limit Enforcement
# ─────────────────────────────────────────────────────────────────

class TestPlanLimitEnforcement:
    """
    Verify that _check_plan_limit raises when memory_counts.total
    exceeds the plan ceiling.
    """

    def _make_db(self, total: int, plan: str) -> MagicMock:
        """Build a mock DB that returns a specific memory count and plan."""
        db = MagicMock()
        # memory_counts query
        counts_result = MagicMock()
        counts_result.data = {"total": total}
        # tenants plan query
        plan_result = MagicMock()
        plan_result.data = {"plan": plan}

        # Chain: db.table(...).select(...).eq(...).single() -> execute()
        db.table.return_value.select.return_value.eq.return_value.single.return_value.execute.side_effect = [
            counts_result,
            plan_result,
        ]
        return db

    @pytest.mark.asyncio
    async def test_free_plan_under_limit_passes(self):
        """Free tenant with 499 memories should not raise."""
        from app.services.memory import _check_plan_limit
        db = self._make_db(total=499, plan="free")
        # Should not raise
        await _check_plan_limit(TENANT_ID, db)

    @pytest.mark.asyncio
    async def test_free_plan_at_limit_raises(self):
        """Free tenant with 500 memories should raise ValueError."""
        from app.services.memory import _check_plan_limit
        db = self._make_db(total=500, plan="free")
        with pytest.raises(ValueError, match="Memory limit reached"):
            await _check_plan_limit(TENANT_ID, db)

    @pytest.mark.asyncio
    async def test_pro_plan_under_limit_passes(self):
        """Pro tenant with 49,999 memories should not raise."""
        from app.services.memory import _check_plan_limit
        db = self._make_db(total=49_999, plan="pro")
        await _check_plan_limit(TENANT_ID, db)

    @pytest.mark.asyncio
    async def test_pro_plan_at_limit_raises(self):
        """Pro tenant with 50,000 memories should raise ValueError."""
        from app.services.memory import _check_plan_limit
        db = self._make_db(total=50_000, plan="pro")
        with pytest.raises(ValueError, match="Memory limit reached"):
            await _check_plan_limit(TENANT_ID, db)

    @pytest.mark.asyncio
    async def test_enterprise_plan_never_raises(self):
        """Enterprise tenant should never hit the limit."""
        from app.services.memory import _check_plan_limit
        db = self._make_db(total=10_000_000, plan="enterprise")
        # Should not raise — enterprise is unlimited
        await _check_plan_limit(TENANT_ID, db)

    @pytest.mark.asyncio
    async def test_no_counts_row_allows_store(self):
        """
        A tenant with no memory_counts row yet (first store ever)
        should be allowed through.
        """
        from app.services.memory import _check_plan_limit
        db = MagicMock()
        no_data_result = MagicMock()
        no_data_result.data = None
        db.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = no_data_result
        # Should not raise — first store path
        await _check_plan_limit(TENANT_ID, db)


# ─────────────────────────────────────────────────────────────────
# Section 4: Refactoring & Architecture Verification
# ─────────────────────────────────────────────────────────────────

class TestBYODClientCaching:
    """Verify BYOD Supabase clients are cached and not recreated on every request."""

    def test_byod_client_caching(self):
        from app.db.client import get_tenant_client
        
        tenant_1 = {
            "mode": "byod",
            "byod_supabase_url": "https://abc123byod.supabase.co",
            "byod_supabase_key": "somekey123"
        }
        tenant_2 = {
            "mode": "byod",
            "byod_supabase_url": "https://abc123byod.supabase.co",
            "byod_supabase_key": "somekey123"
        }
        tenant_different = {
            "mode": "byod",
            "byod_supabase_url": "https://other.supabase.co",
            "byod_supabase_key": "otherkey456"
        }

        with patch("app.db.client.create_client") as mock_create:
            mock_client_a = MagicMock()
            mock_client_b = MagicMock()
            mock_create.side_effect = [mock_client_a, mock_client_b]

            client_1 = get_tenant_client(tenant_1)
            client_2 = get_tenant_client(tenant_2)
            client_diff = get_tenant_client(tenant_different)

            # Identical params should reuse the same cached client
            assert client_1 is client_2
            # Different params should create a new client
            assert client_diff is not client_1
            # create_client should have been called exactly twice
            assert mock_create.call_count == 2


class TestEmbeddingProviderPluggability:
    """Verify pluggable embedding provider selection and initialization."""

    def test_resolve_huggingface_provider(self):
        with patch.dict("os.environ", {"EMBEDDING_PROVIDER": "huggingface", "HUGGINGFACEHUB_API_TOKEN": "hf_test_token"}):
            with patch("app.services.embeddings.InferenceClient") as mock_hf:
                from app.services.embeddings import _resolve_provider, HuggingFaceProvider
                provider = _resolve_provider()
                assert isinstance(provider, HuggingFaceProvider)
                assert provider.model_name == "BAAI/bge-small-en-v1.5"
                assert provider.dimensions == 384

    def test_resolve_openai_provider(self):
        with patch.dict("os.environ", {"EMBEDDING_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test-key"}):
            with patch("app.services.embeddings.OpenAI") as mock_openai:
                from app.services.embeddings import _resolve_provider, OpenAIProvider
                provider = _resolve_provider()
                assert isinstance(provider, OpenAIProvider)
                assert provider.model_name == "text-embedding-3-small"
                assert provider.dimensions == 384


class TestConfigurableSearchWeights:
    """Verify search weights validation and override behavior."""

    def test_weights_validation_success(self):
        from app.schemas.memory import MemorySearch
        # Valid weights summing to 1.0
        payload = MemorySearch(
            query="test", user_id="u1", agent_id="a1",
            similarity_weight=0.5, recency_weight=0.3, importance_weight=0.2
        )
        assert payload.similarity_weight == 0.5
        assert payload.recency_weight == 0.3
        assert payload.importance_weight == 0.2

    def test_weights_validation_incomplete_raises(self):
        from app.schemas.memory import MemorySearch
        from pydantic import ValidationError
        # Missing recency_weight and importance_weight should raise
        with pytest.raises(ValidationError, match="If any search weight is specified"):
            MemorySearch(
                query="test", user_id="u1", agent_id="a1",
                similarity_weight=0.5
            )

    def test_weights_validation_sum_incorrect_raises(self):
        from app.schemas.memory import MemorySearch
        from pydantic import ValidationError
        # Weights summing to 1.2 should raise
        with pytest.raises(ValidationError, match="The sum of similarity_weight"):
            MemorySearch(
                query="test", user_id="u1", agent_id="a1",
                similarity_weight=0.5, recency_weight=0.5, importance_weight=0.2
            )


# ─────────────────────────────────────────────────────────────────
# Section 5: Secondary Audit Verification Tests
# ─────────────────────────────────────────────────────────────────

class TestTenantCreateValidation:
    """Verify TenantCreate schema validates BYOD mode requirements."""

    def test_hosted_mode_passes_without_byod_credentials(self):
        from app.schemas.memory import TenantCreate
        # Should not raise for hosted mode
        tc = TenantCreate(name="Hosted Tenant", email="hosted@example.com", mode="hosted")
        assert tc.mode == "hosted"

    def test_byod_mode_missing_url_raises(self):
        from app.schemas.memory import TenantCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="byod_supabase_url is required"):
            TenantCreate(
                name="BYOD Tenant", email="byod@example.com", mode="byod",
                byod_supabase_key="secretkey"
            )

    def test_byod_mode_missing_key_raises(self):
        from app.schemas.memory import TenantCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="byod_supabase_key is required"):
            TenantCreate(
                name="BYOD Tenant", email="byod@example.com", mode="byod",
                byod_supabase_url="https://supabase.co"
            )

    def test_byod_mode_with_credentials_passes(self):
        from app.schemas.memory import TenantCreate
        tc = TenantCreate(
            name="BYOD Tenant", email="byod@example.com", mode="byod",
            byod_supabase_url="https://supabase.co", byod_supabase_key="secretkey"
        )
        assert tc.mode == "byod"
        assert tc.byod_supabase_url == "https://supabase.co"
        assert tc.byod_supabase_key == "secretkey"


class TestSdkBaseUrlDefaults:
    """Verify synchronous and asynchronous SDK clients default to api.remem.online."""

    def test_sync_client_default_base_url(self):
        from remem import RememClient
        client = RememClient(api_key="rm_fake_key")
        assert client.base_url == "https://api.remem.online"

    def test_async_client_default_base_url(self):
        from remem import AsyncRememClient
        client = AsyncRememClient(api_key="rm_fake_key")
        assert client.base_url == "https://api.remem.online"


class TestMemoryUpdateEndpoint:
    """Verify that updating a memory via PATCH /memories/{memory_id} accepts a body without memory_id."""

    def _get_app(self):
        from app.main import app
        return TestClient(app)

    @patch("app.api.routes.memories.get_current_tenant")
    @patch("app.api.routes.memories.get_tenant_client")
    @patch("app.api.routes.memories.update_memory")
    def test_patch_memory_update_without_id_in_body_passes_validation(self, mock_update, mock_client, mock_tenant):
        mock_tenant.return_value = {"tenant_id": "tenant_test_123"}
        mock_client.return_value = MagicMock()
        
        # Mock successful update response
        from app.schemas.memory import MemoryOut, MemoryType
        mock_update.return_value = MemoryOut(
            id="mem_123",
            content="new content",
            user_id="user_abc",
            agent_id="agent_xyz",
            memory_type="episodic",
            metadata={},
            importance=0.8,
            created_at=datetime.now(),
            last_accessed=None,
            expires_at=None
        )

        client = self._get_app()
        # PATCH request body contains user_id, agent_id, and new_content, but NO memory_id
        resp = client.patch(
            "/memories/mem_123",
            headers={"X-API-Key": "rm_somekey123"},
            json={
                "user_id": "user_abc",
                "agent_id": "agent_xyz",
                "new_content": "new content",
                "importance": 0.8
            }
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == "new content"
        
        # Verify update_memory was called and the constructed MemoryUpdate had memory_id populated from path
        assert mock_update.call_count == 1
        args, kwargs = mock_update.call_args
        payload = kwargs.get("payload") or args[0]
        assert payload.memory_id == "mem_123"
        assert payload.new_content == "new content"
        assert payload.importance == 0.8



