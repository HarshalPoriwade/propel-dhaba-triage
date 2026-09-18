"""Tests verifying Step 1 scaffolding, configuration defaults, and app factory."""

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.main import app, create_app


def test_settings_default_fixture_mode():
    """Verify that MODEL_MODE defaults to 'fixture' and requires no API key."""
    settings = Settings()
    assert settings.MODEL_MODE == "fixture"
    assert settings.MODEL_API_KEY is None
    assert settings.APP_NAME == "Dhaba Support Triage Service"
    assert settings.MODEL_TIMEOUT_SECONDS == 2.5
    assert settings.LLM_MAX_CONCURRENCY == 20
    assert settings.DATABASE_PATH == "dhaba_triage.db"


def test_settings_live_mode_requires_api_key():
    """Verify that MODEL_MODE='live' strictly enforces presence of MODEL_API_KEY."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(MODEL_MODE="live", MODEL_API_KEY=None)
    assert "MODEL_API_KEY is required when MODEL_MODE is set to 'live'" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info_empty:
        Settings(MODEL_MODE="live", MODEL_API_KEY="   ")
    assert "MODEL_API_KEY is required when MODEL_MODE is set to 'live'" in str(exc_info_empty.value)


def test_settings_live_mode_with_api_key():
    """Verify that MODEL_MODE='live' succeeds when an API key is provided."""
    settings = Settings(MODEL_MODE="live", MODEL_API_KEY="test-key-12345")
    assert settings.MODEL_MODE == "live"
    assert settings.MODEL_API_KEY == "test-key-12345"


def test_get_settings_lru_cached():
    """Verify get_settings returns the singleton cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_create_app_factory():
    """Verify create_app produces a valid FastAPI instance with correct metadata."""
    custom_settings = Settings(APP_NAME="Test Triage", APP_ENV="testing")
    application = create_app(custom_settings)
    assert isinstance(application, FastAPI)
    assert application.title == "Test Triage"


def test_exported_app_instance():
    """Verify the module-level exported app is a valid FastAPI instance."""
    assert isinstance(app, FastAPI)
    assert app.title == "Dhaba Support Triage Service"
