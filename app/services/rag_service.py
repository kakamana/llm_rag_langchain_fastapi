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
