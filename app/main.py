# app/main.py
from .create_app import create_app
from fastapi import FastAPI, HTTPException, Query, status
from typing import List, Dict, Optional
import torch
from .models import Question, QuestionResponse  # Updated import
from .services.rag_service import ModelService
import logging
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = create_app()

# Initialize model service
model_service = None

# Error handler for common exceptions
async def handle_exception(exc: Exception) -> JSONResponse:
    logger.error(f"Error occurred: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the model service and load documents on startup"""
    global model_service
    if not app.debug:  # Skip initialization in test mode
        try:
            model_service = ModelService()
            logger.info("Model service initialized successfully!")
            docs_path = "./Docs/All"
            model_service.load_documents(docs_path)
            logger.info("Documents loaded successfully!")
        except Exception as e:
            logger.error(f"Startup error: {str(e)}")
            raise

def check_model_service():
    """Utility function to check if model service is initialized"""
    if not model_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service not initialized"
        )
    return model_service

# Health and System Check Endpoints
@app.get("/health", tags=["System"])
async def health_check():
    """Basic health check endpoint"""
    check_model_service()
    return {
        "status": "healthy",
        "model_loaded": True,
        "documents_loaded": bool(model_service.vectorstore)
    }

@app.get("/system", tags=["System"])
async def system_check():
    """Detailed system status check"""
    check_model_service()
    return {
        "status": "healthy",
        "embeddings_model": str(model_service.embeddings),
        "vectorstore_initialized": bool(model_service.vectorstore),
        "device": str(torch.device("mps" if torch.backends.mps.is_available() else "cpu")),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available()
    }

# Document Management Endpoints
@app.get("/documents", tags=["Documents"], response_model=List[str])
async def list_documents():
    """List all available documents in the system"""
    check_model_service()
    docs = model_service.vectorstore.similarity_search("test", k=100)
    unique_sources = {
        doc.metadata.get('source', 'Unknown Source')
        for doc in docs
        if doc.metadata.get('source') != 'Unknown Source'
    }
    return sorted(list(unique_sources))

@app.get("/documents/pages", tags=["Documents"])
async def document_pages():
    """Get document pages information"""
    check_model_service()
    docs = model_service.vectorstore.similarity_search("test", k=20)
    
    doc_pages = {}
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        doc_pages.setdefault(source, set()).add(page)
    
    return {
        "status": "success",
        "documents": [
            {
                "document": source,
                "pages": sorted(list(pages)),
                "total_pages": len(pages)
            }
            for source, pages in doc_pages.items()
        ]
    }

# Search and QA Endpoints
@app.get("/search", tags=["Search"])
async def search_documents(
    query: str = Query(..., min_length=1, description="Search query to test"),
    k: Optional[int] = Query(
        default=4,
        ge=1,  # greater than or equal to 1
        le=100,  # less than or equal to 100
        description="Number of documents to retrieve"
    )
):
    """Test the document retrieval system"""
    check_model_service()
    try:
        docs = model_service.vectorstore.similarity_search(query, k=k)
        return {
            "query": query,
            "num_results": len(docs),
            "results": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get('source', 'Unknown'),
                    "page": doc.metadata.get('page', 'Unknown')
                }
                for doc in docs
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Update the ask_question endpoint to use QuestionResponse
@app.post("/ask", response_model=QuestionResponse, tags=["Question Answering"])
async def ask_question(question: Question):
    """Ask a question and get an answer with sources"""
    check_model_service()
    try:
        if not question.text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Question text cannot be empty"
            )
        
        response = model_service.get_answer(question.text)
        return QuestionResponse(
            text=response["text"],
            sources=response["sources"]
        )
    except Exception as e:
        return await handle_exception(e)

@app.post("/prompt/test", tags=["Debug"])
async def test_prompt(question: Question):
    """Test the prompt generation"""
    check_model_service()
    docs = model_service.vectorstore.similarity_search(question.text, k=4)
    
    context = "\n\n".join(
        f"Document: {doc.metadata.get('source', 'Unknown')} "
        f"(Page {doc.metadata.get('page', 'Unknown')})\n"
        f"Content: {doc.page_content.strip()}"
        for doc in docs
    )
    
    prompt = f"""Please answer the following question based on the provided document excerpts.
If the answer isn't found in the documents, say "I cannot find specific information about this in the provided documents."

Question: {question.text}

Reference Documents:
{context}

Provide a clear and concise answer (2-3 sentences maximum) using only information from the documents:"""
    
    return {
        "prompt": prompt,
        "num_docs": len(docs),
        "total_context_length": len(context)
    }

# Development server configuration
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )