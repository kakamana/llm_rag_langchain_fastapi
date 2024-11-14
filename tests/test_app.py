# tests/test_app.py
import pytest
from app.models import Question, QuestionResponse, Source

class TestQuestionAnswering:
    def test_ask_question(self, client):
        """Test valid question"""
        question = Question(text="What are the working hours policy?")
        response = client.post("/ask", json=question.dict())
        assert response.status_code == 200
        assert "text" in response.json()
        assert "sources" in response.json()

    def test_invalid_question(self, client):
        """Test error handling for invalid question format"""
        response = client.post("/ask", json={})
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_empty_question(self, client):
        """Test empty question handling"""
        response = client.post("/ask", json={"text": ""})
        assert response.status_code == 422
        assert "detail" in response.json()
        
    def test_whitespace_question(self, client):
        """Test question with only whitespace"""
        response = client.post("/ask", json={"text": "   "})
        assert response.status_code == 422
        assert "detail" in response.json()

class TestSearchEndpoints:
    def test_search(self, client):
        """Test valid search"""
        response = client.get("/search?query=working hours&k=2")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "num_results" in data

    def test_search_with_invalid_k(self, client):
        """Test search with invalid k parameter"""
        response = client.get("/search?query=test&k=-1")
        assert response.status_code == 422
        assert "detail" in response.json()
        
    def test_search_with_empty_query(self, client):
        """Test search with empty query"""
        response = client.get("/search?query=&k=2")
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_search_with_large_k(self, client):
        """Test search with too large k value"""
        response = client.get("/search?query=test&k=1000")
        assert response.status_code == 422
        assert "detail" in response.json()

class TestErrorHandling:
    def test_service_error(self, client, mock_model_service):
        """Test error handling when service fails"""
        mock_model_service.get_answer.side_effect = Exception("Test error")
        question = Question(text="Test question")
        response = client.post("/ask", json=question.dict())
        assert response.status_code == 500
        assert "detail" in response.json()

    def test_response_model_validation(self, client, mock_model_service):
        """Test that response matches QuestionResponse model"""
        # Reset the mock to return valid data
        mock_model_service.get_answer.side_effect = None
        mock_model_service.get_answer.return_value = {
            "text": "This is a test answer",
            "sources": [{"document": "test.pdf", "page": "1"}]
        }
        
        question = Question(text="Test question")
        response = client.post("/ask", json=question.dict())
        assert response.status_code == 200
        
        # Validate response structure
        response_data = response.json()
        question_response = QuestionResponse(**response_data)
        assert isinstance(question_response.text, str)
        assert len(question_response.sources) > 0