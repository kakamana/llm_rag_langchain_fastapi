# tests/conftest.py
import sys
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Import after path setup
from app.main import app
from app.services.rag_service import ModelService

@pytest.fixture(scope="session")
def mock_model_service():
    """Create a mock model service"""
    mock_service = MagicMock(spec=ModelService)
    
    # Mock vectorstore
    mock_service.vectorstore = MagicMock()
    mock_service.vectorstore.similarity_search.return_value = [
        MagicMock(
            page_content="Sample content",
            metadata={"source": "test.pdf", "page": "1"}
        )
    ]
    
    # Mock embeddings
    mock_service.embeddings = MagicMock()
    mock_service.embeddings.__str__.return_value = "mock_embeddings"
    
    # Mock get_answer method
    mock_service.get_answer.return_value = {
        "text": "This is a test answer",
        "sources": [{"document": "test.pdf", "page": "1"}]
    }
    
    return mock_service

@pytest.fixture
def test_app(mock_model_service):
    """Create test app with mocked dependencies"""
    # Set up test dependencies
    import app.main
    app.main.model_service = mock_model_service
    return app.main.app

@pytest.fixture
def client(test_app):
    """Create test client"""
    return TestClient(test_app)

@pytest.fixture(scope="session")
def mock_docs_dir(tmp_path_factory):
    """Create temporary directory for test documents"""
    tmp_path = tmp_path_factory.mktemp("test_docs")
    docs_dir = tmp_path / "Docs" / "All"
    docs_dir.mkdir(parents=True)
    return str(docs_dir)