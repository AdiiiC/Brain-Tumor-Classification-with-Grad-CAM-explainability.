"""API key auth and rate limiting."""

from __future__ import annotations

from api.security import SlidingWindowRateLimiter, auth_enabled


class TestAuthConfiguration:
    def test_auth_disabled_when_no_keys(self, monkeypatch):
        monkeypatch.delenv("API_KEYS", raising=False)
        assert auth_enabled() is False

    def test_auth_enabled_when_keys_present(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "abc123")
        assert auth_enabled() is True

    def test_blank_keys_do_not_enable_auth(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", " , ,")
        assert auth_enabled() is False


class TestAuthEnforcement:
    def test_missing_key_rejected(self, client, monkeypatch, upload):
        monkeypatch.setenv("API_KEYS", "secret-key")
        response = client.post("/predict", files=upload())
        assert response.status_code == 401

    def test_wrong_key_rejected(self, client, monkeypatch, upload):
        monkeypatch.setenv("API_KEYS", "secret-key")
        response = client.post("/predict", files=upload(), headers={"X-API-Key": "wrong"})
        assert response.status_code == 403

    def test_correct_key_passes_auth(self, client, monkeypatch, upload):
        monkeypatch.setenv("API_KEYS", "secret-key")
        response = client.post("/predict", files=upload(), headers={"X-API-Key": "secret-key"})
        # Auth passes; the model is absent in CI so 503 is the expected outcome.
        assert response.status_code == 503

    def test_multiple_keys_supported(self, client, monkeypatch, upload):
        monkeypatch.setenv("API_KEYS", "key-one,key-two")
        response = client.post("/predict", files=upload(), headers={"X-API-Key": "key-two"})
        assert response.status_code == 503

    def test_health_is_public(self, client, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key")
        assert client.get("/health").status_code == 200


class TestRateLimiter:
    def test_allows_up_to_limit(self):
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            allowed, _ = limiter.check("client")
            assert allowed

    def test_blocks_beyond_limit(self):
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        limiter.check("client")
        limiter.check("client")
        allowed, retry_after = limiter.check("client")
        assert allowed is False
        assert retry_after > 0

    def test_clients_are_isolated(self):
        limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        assert limiter.check("a")[0] is True
        assert limiter.check("b")[0] is True
        assert limiter.check("a")[0] is False

    def test_window_expiry_releases_budget(self):
        limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=0)
        assert limiter.check("a")[0] is True
        assert limiter.check("a")[0] is True

    def test_returns_429_when_enabled(self, client, monkeypatch, upload):
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        from api.security import reset_all_limiters
        reset_all_limiters()

        statuses = {client.post("/analyze", files=upload()).status_code for _ in range(14)}
        assert 429 in statuses


class TestCORS:
    def test_wildcard_is_not_used(self, client):
        response = client.options(
            "/health",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
        )
        assert response.headers.get("access-control-allow-origin") != "*"

    def test_configured_origin_allowed(self, client):
        response = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
