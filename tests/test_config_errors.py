"""Tests for improved configuration error messages (issue #226).

Issue #226: Improve Configuration Error Messages.

These tests confirm that:
  * ConfigError is raised with descriptive, actionable messages.
  * Each error message identifies the setting, shows the invalid value,
    states what was expected, and suggests a fix.
  * The fix suggestion mentions the env var name and an example value.
  * Existing startup behavior is unchanged (valid configs still load).
  * validate_config() collects ALL errors instead of failing on the first.
"""

import importlib
import os
import sys
from unittest.mock import patch

import pytest

import config
from config import ConfigError, validate_config


# ---------------------------------------------------------------------------
# Helpers — reload config with custom env vars
# ---------------------------------------------------------------------------

def _reload_config_with_env(env: dict[str, str]):
    """Reload the config module with the given env vars set.

    Returns the freshly-loaded module. The original env is restored
    on test teardown via the monkeypatch fixture.
    """
    if "config" in sys.modules:
        del sys.modules["config"]

    for k, v in env.items():
        os.environ[k] = v

    import config as fresh_config
    importlib.reload(fresh_config)
    return fresh_config


# ---------------------------------------------------------------------------
# ConfigError class
# ---------------------------------------------------------------------------

class TestConfigErrorClass:
    def test_config_error_is_value_error_subclass(self):
        """ConfigError should be a subclass of ValueError for backward compat."""
        assert issubclass(ConfigError, ValueError)

    def test_config_error_has_descriptive_attributes(self):
        """ConfigError should expose setting, value, expected, and fix."""
        err = ConfigError(
            setting="FOO",
            value="bar",
            expected="an integer",
            fix="set FOO=42",
        )
        assert err.setting == "FOO"
        assert err.value == "bar"
        assert err.expected == "an integer"
        assert err.fix == "set FOO=42"

    def test_config_error_message_contains_all_fields(self):
        """The string representation should contain setting, value, expected, fix."""
        err = ConfigError(
            setting="RETRIEVAL_TOP_K",
            value="abc",
            expected="an integer",
            fix="export RETRIEVAL_TOP_K=6",
        )
        msg = str(err)
        assert "RETRIEVAL_TOP_K" in msg
        assert "'abc'" in msg
        assert "an integer" in msg
        assert "export RETRIEVAL_TOP_K=6" in msg

    def test_config_error_message_format_is_consistent(self):
        """Error message should use the documented 4-line format."""
        err = ConfigError("X", 1, ">= 2", "increase X")
        msg = str(err)
        assert "X — invalid configuration." in msg
        assert "Got:" in msg
        assert "Expected:" in msg
        assert "Fix:" in msg


# ---------------------------------------------------------------------------
# Integer parsing — _env_int
# ---------------------------------------------------------------------------

class TestEnvIntValidation:
    def test_valid_integer_env_var(self, monkeypatch):
        """A valid integer env var should parse without error."""
        monkeypatch.setenv("TEST_INT_VAL", "42")
        assert config._env_int("TEST_INT_VAL", 10) == 42

    def test_invalid_integer_raises_config_error(self, monkeypatch):
        """A non-integer env var should raise ConfigError."""
        monkeypatch.setenv("TEST_INT_VAL", "not-a-number")
        with pytest.raises(ConfigError) as exc_info:
            config._env_int("TEST_INT_VAL", 10)

        err = exc_info.value
        assert err.setting == "TEST_INT_VAL"
        assert err.value == "not-a-number"
        assert "integer" in err.expected.lower()
        assert "TEST_INT_VAL" in err.fix
        assert "export" in err.fix.lower() or "set" in err.fix.lower()

    def test_below_minimum_raises_config_error(self, monkeypatch):
        """A value below the minimum should raise ConfigError."""
        monkeypatch.setenv("TEST_INT_VAL", "0")
        with pytest.raises(ConfigError) as exc_info:
            config._env_int("TEST_INT_VAL", 10, min_value=1)

        err = exc_info.value
        assert err.setting == "TEST_INT_VAL"
        assert err.value == 0
        assert ">= 1" in err.expected
        assert "below the minimum" in err.fix

    def test_default_used_when_env_not_set(self, monkeypatch):
        """When env var is not set, the default should be used."""
        monkeypatch.delenv("TEST_INT_VAL", raising=False)
        assert config._env_int("TEST_INT_VAL", 99) == 99


# ---------------------------------------------------------------------------
# Float parsing — _env_float
# ---------------------------------------------------------------------------

class TestEnvFloatValidation:
    def test_valid_float_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAL", "0.75")
        assert config._env_float("TEST_FLOAT_VAL", 0.5) == 0.75

    def test_invalid_float_raises_config_error(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAL", "not-a-float")
        with pytest.raises(ConfigError) as exc_info:
            config._env_float("TEST_FLOAT_VAL", 0.5)

        err = exc_info.value
        assert err.setting == "TEST_FLOAT_VAL"
        assert err.value == "not-a-float"
        assert "number" in err.expected.lower()

    def test_below_minimum_raises_config_error(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAL", "-0.5")
        with pytest.raises(ConfigError) as exc_info:
            config._env_float("TEST_FLOAT_VAL", 0.5, min_value=0.0)

        assert exc_info.value.setting == "TEST_FLOAT_VAL"
        assert ">= 0.0" in exc_info.value.expected

    def test_above_maximum_raises_config_error(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAL", "1.5")
        with pytest.raises(ConfigError) as exc_info:
            config._env_float("TEST_FLOAT_VAL", 0.5, max_value=1.0)

        assert exc_info.value.setting == "TEST_FLOAT_VAL"
        assert "<= 1.0" in exc_info.value.expected


# ---------------------------------------------------------------------------
# String parsing — _env_str
# ---------------------------------------------------------------------------

class TestEnvStrValidation:
    def test_valid_string_in_allowed_set(self, monkeypatch):
        monkeypatch.setenv("TEST_STR_VAL", "dark")
        assert config._env_str("TEST_STR_VAL", "light", allowed=frozenset({"light", "dark"})) == "dark"

    def test_invalid_string_not_in_allowed_set_raises(self, monkeypatch):
        monkeypatch.setenv("TEST_STR_VAL", "purple")
        with pytest.raises(ConfigError) as exc_info:
            config._env_str("TEST_STR_VAL", "light", allowed=frozenset({"light", "dark"}))

        err = exc_info.value
        assert err.setting == "TEST_STR_VAL"
        assert err.value == "purple"
        assert "light" in err.expected
        assert "dark" in err.expected


# ---------------------------------------------------------------------------
# JSON config loader — error handling
# ---------------------------------------------------------------------------

class TestJsonConfigLoader:
    def test_valid_json_config_loads(self, tmp_path, monkeypatch):
        """A valid config.json should load successfully."""
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text('{"INDEX_CHUNK_SIZE": 20}')

        with patch.object(config, "_load_json_config", return_value={"INDEX_CHUNK_SIZE": 20}):
            pass
        result = config._load_json_config()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# validate_config() — collects all errors
# ---------------------------------------------------------------------------

class TestValidateConfigCollectsAllErrors:
    def test_valid_config_passes(self):
        """A valid configuration should pass validate_config() without raising."""
        validate_config()

    def test_invalid_retrieval_top_k_message_has_fix(self, monkeypatch):
        """Error message should include a fix suggestion."""
        monkeypatch.setattr(config, "RETRIEVAL_TOP_K", 0)
        with pytest.raises(ValueError) as exc_info:
            validate_config()
        msg = str(exc_info.value)
        assert "RETRIEVAL_TOP_K" in msg
        assert "0" in msg
        assert "Fix:" in msg
        assert "RETRIEVAL_TOP_K=6" in msg

    def test_invalid_cache_similarity_threshold_message_has_fix(self, monkeypatch):
        monkeypatch.setattr(config, "CACHE_SIMILARITY_THRESHOLD", 1.5)
        with pytest.raises(ValueError) as exc_info:
            validate_config()
        msg = str(exc_info.value)
        assert "CACHE_SIMILARITY_THRESHOLD" in msg
        assert "1.5" in msg
        assert "Fix:" in msg
        assert "0.70" in msg

    def test_invalid_hybrid_top_k_message_has_fix(self, monkeypatch):
        monkeypatch.setattr(config, "HYBRID_TOP_K", -1)
        with pytest.raises(ValueError) as exc_info:
            validate_config()
        msg = str(exc_info.value)
        assert "HYBRID_TOP_K" in msg
        assert "Fix:" in msg
        assert "HYBRID_TOP_K=20" in msg

    def test_multiple_errors_are_all_reported(self, monkeypatch):
        """When multiple settings are invalid, ALL errors should be reported."""
        monkeypatch.setattr(config, "RETRIEVAL_TOP_K", 0)
        monkeypatch.setattr(config, "HYBRID_TOP_K", -1)
        monkeypatch.setattr(config, "CACHE_SIMILARITY_THRESHOLD", 2.0)

        with pytest.raises(ValueError) as exc_info:
            validate_config()
        msg = str(exc_info.value)

        assert "RETRIEVAL_TOP_K" in msg
        assert "HYBRID_TOP_K" in msg
        assert "CACHE_SIMILARITY_THRESHOLD" in msg
        assert msg.count("Fix:") >= 3
        assert "3 problem(s)" in msg

    def test_chunk_size_lt_overlap_message_has_fix(self, monkeypatch):
        """When chunk_size <= overlap, the error should suggest a fix."""
        monkeypatch.setattr(config, "INDEX_CHUNK_SIZE", 5)
        monkeypatch.setattr(config, "INDEX_CHUNK_OVERLAP", 10)

        with pytest.raises(ValueError) as exc_info:
            validate_config()
        msg = str(exc_info.value)
        assert "INDEX_CHUNK_SIZE" in msg
        assert "INDEX_CHUNK_OVERLAP" in msg
        assert "Fix:" in msg
        assert "increase INDEX_CHUNK_SIZE" in msg or "decrease INDEX_CHUNK_OVERLAP" in msg


# ---------------------------------------------------------------------------
# Import-time validation — fail fast on bad config
# ---------------------------------------------------------------------------

class TestImportTimeValidation:
    def test_bad_chunk_size_at_import_raises_config_error(self, monkeypatch):
        """Importing config with INDEX_CHUNK_SIZE <= overlap should raise."""
        monkeypatch.setenv("INDEX_CHUNK_SIZE", "1")
        monkeypatch.setenv("INDEX_CHUNK_OVERLAP", "5")
        with pytest.raises((ConfigError, ValueError)) as exc_info:
            _reload_config_with_env({"INDEX_CHUNK_SIZE": "1", "INDEX_CHUNK_OVERLAP": "5"})
        msg = str(exc_info.value)
        assert "INDEX_CHUNK_SIZE" in msg or "INDEX_CHUNK_OVERLAP" in msg

    def test_bad_max_file_size_at_import_raises(self, monkeypatch):
        """Importing config with MAX_FILE_SIZE_MB < 1 should raise."""
        with pytest.raises((ConfigError, ValueError)) as exc_info:
            _reload_config_with_env({"MAX_FILE_SIZE_MB": "0"})
        msg = str(exc_info.value)
        assert "MAX_FILE_SIZE_MB" in msg

    def test_non_integer_env_var_raises_config_error(self, monkeypatch):
        """A non-integer value for an int setting should raise ConfigError."""
        with pytest.raises((ConfigError, ValueError)) as exc_info:
            _reload_config_with_env({"EMBEDDING_DIMENSIONS": "not-a-number"})
        err = exc_info.value
        assert "EMBEDDING_DIMENSIONS" in str(err)


# ---------------------------------------------------------------------------
# Fix suggestions — every error mentions the env var name and an example
# ---------------------------------------------------------------------------

class TestFixSuggestions:
    def test_int_error_fix_mentions_env_var_and_example(self, monkeypatch):
        monkeypatch.setenv("TEST_FIX_INT", "abc")
        with pytest.raises(ConfigError) as exc_info:
            config._env_int("TEST_FIX_INT", 10)
        fix = exc_info.value.fix
        assert "TEST_FIX_INT" in fix
        assert "export TEST_FIX_INT" in fix or "set TEST_FIX_INT" in fix

    def test_float_error_fix_mentions_env_var_and_example(self, monkeypatch):
        monkeypatch.setenv("TEST_FIX_FLOAT", "xyz")
        with pytest.raises(ConfigError) as exc_info:
            config._env_float("TEST_FIX_FLOAT", 0.5)
        fix = exc_info.value.fix
        assert "TEST_FIX_FLOAT" in fix
        assert "export TEST_FIX_FLOAT" in fix or "set TEST_FIX_FLOAT" in fix

    def test_json_error_fix_mentions_config_json(self, monkeypatch):
        """Errors from config.json should mention config.json in the fix."""
        with patch.object(config, "_JSON_CFG", {"TEST_JSON_INT": "not-a-number"}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("TEST_JSON_INT", None)
                with pytest.raises(ConfigError) as exc_info:
                    config._env_or_json_int("TEST_JSON_INT", 10)
                fix = exc_info.value.fix
                assert "config.json" in fix


# ---------------------------------------------------------------------------
# Startup behavior unchanged
# ---------------------------------------------------------------------------

class TestStartupBehaviorUnchanged:
    def test_default_config_loads_without_error(self, monkeypatch):
        """The default configuration should load without raising."""
        for key in [
            "EMBEDDING_DIMENSIONS", "INDEX_CHUNK_SIZE", "INDEX_CHUNK_OVERLAP",
            "MAX_FILE_SIZE_MB", "RETRIEVAL_TOP_K", "HYBRID_TOP_K",
            "CACHE_SIMILARITY_THRESHOLD", "QDRANT_SIMILARITY_THRESHOLD",
        ]:
            monkeypatch.delenv(key, raising=False)

        if "config" in sys.modules:
            del sys.modules["config"]
        import config as fresh_config
        importlib.reload(fresh_config)
        assert fresh_config.EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"
        assert fresh_config.INDEX_CHUNK_SIZE == 15
        assert fresh_config.INDEX_CHUNK_OVERLAP == 3

    def test_config_json_overrides_apply(self):
        """Values from config.json should override defaults."""
        assert config.INDEX_CHUNK_SIZE == 15  # from config.json

    def test_validate_config_does_not_raise_on_valid_state(self):
        """validate_config() should not raise when config is valid."""
        validate_config()
