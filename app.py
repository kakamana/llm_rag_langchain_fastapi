# app.py
'''
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List, Dict, Optional
import logging
# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app = FastAPI()

class DocumentSource(BaseModel):
    document: str
    page: str

class QuestionResponse(BaseModel):
    text: str = Field(..., description="The answer to the question")
    sources: List[DocumentSource] = Field(..., description="Sources used for the answer")

class Question(BaseModel):
    text: str

class Source(BaseModel):
    document: str
    page: str

class Answer(BaseModel):
    text: str
    sources: List[Source]

class ModelService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.vectorstore = None
        self.embeddings = None
        self.initialize_model()
        
    def initialize_model(self):
        try:
            # Initialize language model
            logger.info("Loading Phi-2 model...")
            model_id = "microsoft/phi-2"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                trust_remote_code=True
            )
            
            # Configure model loading for M2 Mac
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                load_in_8bit=False,  # Changed for M2 compatibility
                device_map="auto",
                # Add M2-specific configurations
                use_cache=True,
                low_cpu_mem_usage=True
            )
            
            # Initialize embeddings specifically for M2
            logger.info("Initializing embeddings model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={
                    'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
                }
            )
            logger.info("Model initialization complete!")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def load_documents(self, pdf_folder: str):
        try:
            logger.info(f"Loading documents from: {pdf_folder}")
            documents = []
            
            if not os.path.exists(pdf_folder):
                raise Exception(f"Folder not found: {pdf_folder}")
            
            pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            for filename in pdf_files:
                try:
                    file_path = os.path.join(pdf_folder, filename)
                    logger.info(f"Processing: {filename}")
                    
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    
                    # Properly set page numbers in metadata
                    for i, page in enumerate(pages, start=1):
                        page.metadata.update({
                            'source': filename,
                            'page': str(i)  # Convert to string for consistency
                        })
                    
                    documents.extend(pages)
                    logger.info(f"Loaded {len(pages)} pages from {filename}")
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
            
            logger.info(f"Total pages loaded: {len(documents)}")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=250,
                chunk_overlap=50  # Added some overlap for better context
            )
            splits = text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} text chunks")
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings
            )
            logger.info("Vector store creation complete!")
            
        except Exception as e:
            logger.error(f"Error in load_documents: {str(e)}")
            raise

    
    def get_answer(self, question: str) -> Dict:
        try:
            logger.info(f"Processing question: {question}")
            
            # Get relevant documents
            docs = self.vectorstore.similarity_search(question, k=4)
            logger.info(f"Found {len(docs)} relevant documents")
            
            # Format context
            context_parts = []
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                content = doc.page_content.strip()
                context_parts.append(f"Document: {source} (Page {page})\nContent: {content}")
            
            context = "\n\n".join(context_parts)
            
            # Simplified prompt
            prompt = f"""Please answer the following question based on the provided document excerpts.
    If the answer isn't found in the documents, say "I cannot find specific information about this in the provided documents."

    Question: {question}

    Reference Documents:
    {context}

    Provide a clear and concise answer (2-3 sentences maximum) using only information from the documents:"""
            
            # Generate answer
            logger.info("Generating answer...")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,  # Reduced for more concise answers
                temperature=0.3,     # Lower temperature for more focused responses
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Get the generated response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response
            def clean_response(response: str) -> str:
                # Remove the prompt parts
                if "Provide a clear and concise answer" in response:
                    response = response.split("Provide a clear and concise answer")[-1]
                
                # Remove any remaining prompt parts
                if "Question:" in response:
                    response = response.split("Question:")[-1]
                if "Reference Documents:" in response:
                    response = response.split("Reference Documents:")[0]
                    
                # Clean up whitespace and special characters
                response = response.strip()
                response = response.replace('\n', ' ')
                response = ' '.join(response.split())
                
                return response
            
            final_answer = clean_response(response)
            
            # Format the final response with sources
            source_citations = [
                f"[{doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', 'Unknown')}]"
                for doc in docs
            ]
            
            final_response = f"{final_answer}\n\nSources: {', '.join(source_citations)}"
            
            return {
                "text": final_response,
                "sources": [
                    {
                        "document": doc.metadata.get('source', 'Unknown'),
                        "page": doc.metadata.get('page', 'Unknown')
                    }
                    for doc in docs
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in get_answer: {str(e)}")
            raise

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
    '''