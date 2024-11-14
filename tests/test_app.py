# tests/test_app.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ask_question():
    question = {
        "text": "What are the working hours policy?"
    }
    response = client.post("/ask", json=question)
    assert response.status_code == 200
    assert "text" in response.json()
    assert "sources" in response.json()

# Run tests with:
# pytest tests/test_app.py -v