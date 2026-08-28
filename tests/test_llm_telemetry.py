import pytest
from unittest.mock import MagicMock, patch
from llm import get_llm_provider_info, record_telemetry, generate_response, generate_response_stream
from fastapi.testclient import TestClient
from main import app

def test_get_llm_provider_info():
    info = get_llm_provider_info()
    assert "provider" in info
    assert "model" in info
    assert "base_url" in info
    assert "total_requests" in info
    assert isinstance(info["recent_telemetry"], list)

def test_telemetry_recording():
    record_telemetry({"test_key": "test_val", "model": "test-model"})
    info = get_llm_provider_info()
    assert info["total_requests"] >= 1
    assert any(rec.get("test_key") == "test_val" for rec in info["recent_telemetry"])

def test_llm_info_and_telemetry_endpoints():
    client = TestClient(app)
    
    # Test /llm/info
    res_info = client.get("/llm/info")
    assert res_info.status_code == 200
    data_info = res_info.json()
    assert "provider" in data_info
    assert "model" in data_info

    # Test /llm/telemetry
    res_telem = client.get("/llm/telemetry")
    assert res_telem.status_code == 200
    data_telem = res_telem.json()
    assert data_telem["status"] == "ok"
    assert "recent_telemetry" in data_telem

def test_generate_response_stream_telemetry():
    with patch("llm._get_client") as mock_client:
        mock_choice = MagicMock()
        mock_choice.delta.content = "token1 "
        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        
        mock_client.return_value.chat.completions.create.return_value = [mock_chunk]
        
        tokens = list(generate_response_stream("test query", "code context"))
        assert tokens == ["token1 "]
        
        info = get_llm_provider_info()
        recent = info["recent_telemetry"]
        assert any(r.get("type") == "stream" and r.get("token_chunks") == 1 for r in recent)
