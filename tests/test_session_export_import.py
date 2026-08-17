import pytest
from session_manager import SessionManager
from fastapi.testclient import TestClient
from main import app

def test_session_manager_export_import_delete():
    sm = SessionManager(max_sessions=10, max_history_per_session=5)
    sm.update("session-1", "Hello", "Hi there!")
    sm.update("session-1", "How does indexing work?", "It parses python code into AST chunks.")
    
    # Test export
    exported = sm.export_session("session-1")
    assert exported is not None
    assert exported["session_id"] == "session-1"
    assert exported["message_count"] == 2
    assert len(exported["history"]) == 2
    assert "User: Hello" in exported["history"][0]

    # Test non-existent export
    assert sm.export_session("non-existent") is None

    # Test delete
    assert sm.delete_session("session-1") is True
    assert sm.get("session-1") == ""
    assert sm.delete_session("session-1") is False

    # Test import
    imported_id = sm.import_session(exported)
    assert imported_id == "session-1"
    history = sm.get("session-1")
    assert "User: Hello" in history
    assert "User: How does indexing work?" in history


def test_session_manager_import_validation():
    sm = SessionManager()
    with pytest.raises(ValueError, match="Session data must be a valid dictionary"):
        sm.import_session("not-a-dict")

    with pytest.raises(ValueError, match="Session data must contain a non-empty string 'session_id'"):
        sm.import_session({"session_id": ""})


def test_history_endpoints_export_import_delete():
    client = TestClient(app)

    # Populate a session
    client.post("/reset")
    from main import session_manager
    session_manager.update("test-session-42", "What is FastAPI?", "FastAPI is a modern web framework.")

    # Test export endpoint
    export_resp = client.get("/history/export/test-session-42")
    assert export_resp.status_code == 200
    export_data = export_resp.json()
    assert export_data["status"] == "ok"
    assert export_data["session"]["session_id"] == "test-session-42"
    assert len(export_data["session"]["history"]) == 1

    # Test export 404
    export_404 = client.get("/history/export/unknown-sess")
    assert export_404.status_code == 404

    # Test delete endpoint
    del_resp = client.delete("/history/test-session-42")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # Verify session is gone
    get_resp = client.get("/history?session_id=test-session-42")
    assert get_resp.status_code == 200
    assert get_resp.json()["history"] == []

    # Test delete 404
    del_404 = client.delete("/history/test-session-42")
    assert del_404.status_code == 404

    # Test import endpoint
    import_resp = client.post("/history/import", json={"session": export_data["session"]})
    assert import_resp.status_code == 200
    assert import_resp.json()["status"] == "imported"
    assert import_resp.json()["session_id"] == "test-session-42"

    # Verify re-imported session history
    recheck_resp = client.get("/history?session_id=test-session-42")
    assert len(recheck_resp.json()["history"]) == 1
    assert "FastAPI" in recheck_resp.json()["history"][0]
