# app/create_app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def create_app(testing=False):
    app = FastAPI(
        title="RAG Q&A API",
        description="API for question answering using RAG with document sources",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    if testing:
        app.debug = True

    return app