"""Unit tests for config.py.  Run: python -m pytest test_config.py -v"""
from __future__ import annotations

import pytest
from config import Settings, get_settings
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # strip anything that could leak in from the real shell / .env
    for key in list(__import__("os").environ):
        if key.startswith("TRIAGE_"):
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _settings(**env):
    return Settings(_env_file=None, **env)


class TestDefaults:
    def test_defaults(self):
        s = _settings()
        assert s.model == "gemini-2.5-flash"
        assert s.max_tokens == 256
        assert s.environment == "dev"
        assert s.log_json is True
        assert s.database_url == "sqlite:///./triage.db"
        assert s.system_prompt_path is None
        assert s.system_prompt_override() is None

    def test_frozen(self):
        s = _settings()
        with pytest.raises(ValidationError):
            s.model = "claude-opus-5"


class TestEnvParsing:
    def test_env_prefix(self, monkeypatch):
        monkeypatch.setenv("TRIAGE_MODEL", "claude-sonnet-5")
        monkeypatch.setenv("TRIAGE_MAX_TOKENS", "512")
        s = Settings()
        assert s.model == "claude-sonnet-5"
        assert s.max_tokens == 512

    def test_unknown_var_ignored(self, monkeypatch):
        monkeypatch.setenv("TRIAGE_NONSENSE", "x")
        _settings()  # must not raise

    def test_bad_log_level(self, monkeypatch):
        monkeypatch.setenv("TRIAGE_LOG_LEVEL", "verbose")
        with pytest.raises(ValidationError):
            Settings()

    @pytest.mark.parametrize("value", ["0", "999999", "-1"])
    def test_max_tokens_bounds(self, monkeypatch, value):
        monkeypatch.setenv("TRIAGE_MAX_TOKENS", value)
        with pytest.raises(ValidationError):
            Settings()


class TestValidators:
    def test_empty_db_url_rejected(self):
        with pytest.raises(ValidationError):
            _settings(database_url="   ")

    def test_missing_prompt_path_rejected(self):
        with pytest.raises(ValidationError):
            _settings(system_prompt_path="does/not/exist.txt")

    def test_prompt_path_ok_and_readable(self, tmp_path):
        p = tmp_path / "prompt.txt"
        p.write_text("you are a triage engine", encoding="utf-8")
        s = _settings(system_prompt_path=str(p))
        assert s.system_prompt_override() == "you are a triage engine"

    def test_prod_requires_json_logs(self):
        with pytest.raises(ValidationError):
            _settings(environment="prod", log_json=False,
                      database_url="postgresql://localhost/triage")

    def test_prod_rejects_sqlite(self):
        with pytest.raises(ValidationError):
            _settings(environment="prod", database_url="sqlite:///./triage.db")

    def test_prod_ok_with_postgres_and_json(self):
        s = _settings(environment="prod",
                      database_url="postgresql://localhost/triage")
        assert s.is_prod is True


class TestSingleton:
    def test_cached(self):
        assert get_settings() is get_settings()

    def test_cache_clear_reloads(self, monkeypatch):
        first = get_settings()
        monkeypatch.setenv("TRIAGE_SERVICE_NAME", "other")
        get_settings.cache_clear()
        assert get_settings().service_name == "other"
        assert get_settings() is not first
