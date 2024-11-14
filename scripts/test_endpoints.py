# scripts/test_endpoints.py
import requests
import json

def test_endpoints():
    BASE_URL = "http://localhost:8000"
    
    # Test health
    health_response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", health_response.json())
    
    # Test question
    question = {
        "text": "What are the working hours policy?"
    }
    qa_response = requests.post(
        f"{BASE_URL}/ask",
        json=question
    )
    print("\nQuestion Response:", json.dumps(qa_response.json(), indent=2))
    
    # Test documents list (if implemented)
    docs_response = requests.get(f"{BASE_URL}/test/documents")
    print("\nAvailable Documents:", docs_response.json())

if __name__ == "__main__":
    test_endpoints()

# Run with:
# python scripts/test_endpoints.py