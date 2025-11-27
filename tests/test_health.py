from fastapi.testclient import TestClient
from scripts.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    payload = r.json()
    assert "ok" in str(payload.get("status", "")).lower() or isinstance(payload.get("model_loaded"), bool)