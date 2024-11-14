# RAG with LangChain and FastAPI

A Question-Answering system using RAG (Retrieval Augmented Generation) with LangChain and FastAPI.

## Setup

1. Clone the repository
```bash
git clone <llm_rag_langchain_fastapi>
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Install Ollama (Optional - if using Ollama)
- Visit https://ollama.ai/download
- Download and install for your platform

5. Set up your documents
- Create a `Docs/All` directory
- Place your PDF documents in this directory

## Running the Application

1. Start the FastAPI server
```bash
uvicorn app:app --reload
```

2. Access the API documentation
- Open http://localhost:8000/docs in your browser

## API Endpoints

- `POST /ask`: Ask questions about your documents
- `GET /health`: Check system health
- `GET /test/documents`: List available documents

## Project Structure

```
project/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── Docs/              # Document directory
    └── All/           # PDF documents
```

## Environment Variables (if needed)
```env
OPENAI_API_KEY=your_key_here
```