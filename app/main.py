from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Optional
import torch
from models import QuestionResponse, Question
import logging
# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

from app.services.rag_service import ModelService  # Make sure to import the ModelService class

# Initialize model service
model_service = ModelService()

@app.on_event("startup")
async def startup_event():
    # Load documents (update path as needed)
    model_service.load_documents("Docs/All")

@app.get("/test/documents", response_model=List[str])
async def list_documents():
    """List all available documents in the system"""
    try:
        # Get a sample of documents to extract unique sources
        docs = model_service.vectorstore.similarity_search("test", k=100)
        unique_sources = set()
        
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown Source')
            if source != 'Unknown Source':
                unique_sources.add(source)
        
        return sorted(list(unique_sources))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/search")
async def test_search(
    query: str = Query(..., description="Search query to test"),
    k: Optional[int] = Query(4, description="Number of documents to retrieve")
):
    """Test the document retrieval system"""
    try:
        docs = model_service.vectorstore.similarity_search(query, k=k)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page_number', 'Unknown')
            })
        
        return {
            "query": query,
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a more detailed health check
@app.get("/test/system")
async def system_check():
    """Detailed system status check"""
    try:
        return {
            "status": "healthy",
            "model_loaded": model_service is not None,
            "embeddings_model": str(model_service.embeddings),
            "vectorstore_initialized": model_service.vectorstore is not None,
            "device": torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/document-pages")
async def test_document_pages():
    """Test endpoint to verify document page numbers"""
    try:
        # Get a sample of documents
        docs = model_service.vectorstore.similarity_search("test", k=20)
        
        # Group by document
        doc_pages = {}
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            
            if source not in doc_pages:
                doc_pages[source] = set()
            doc_pages[source].add(page)
        
        # Format results
        results = []
        for source, pages in doc_pages.items():
            results.append({
                "document": source,
                "pages": sorted(list(pages)),
                "total_pages": len(pages)
            })
        
        return {
            "status": "success",
            "documents": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(question: Question):
    """
    Ask a question about the loaded documents.
    
    The response will include:
    - A concise answer based on the document content
    - List of sources used (document names and page numbers)
    """
    try:
        if not model_service:
            raise HTTPException(status_code=500, detail="Model service not initialized")
        response = model_service.get_answer(question.text)
        return response
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add test endpoint to verify prompt and response
@app.post("/test/prompt")
async def test_prompt(question: Question):
    """Test endpoint to see the full prompt being sent to the model"""
    try:
        docs = model_service.vectorstore.similarity_search(question.text, k=4)
        context_parts = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            content = doc.page_content.strip()
            context_parts.append(f"Document: {source} (Page {page})\nContent: {content}")
        
        context = "\n\n".join(context_parts)
        
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)