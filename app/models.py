# app/models.py
from pydantic import BaseModel, Field, validator
from typing import List

class Source(BaseModel):
    """Document source with page information"""
    document: str = Field(..., description="Source document name")
    page: str = Field(..., description="Page number in the document")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document": "employee_handbook.pdf",
                "page": "5"
            }
        }

class Question(BaseModel):
    """Input question model"""
    text: str = Field(..., min_length=1, description="The question to ask")
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Question text cannot be empty')
        return v.strip()
    class Config:
        json_schema_extra = {
            "example": {
                "text": "What are the working hours policy?"
            }
        }

class QuestionResponse(BaseModel):
    """Response model with answer and sources"""
    text: str = Field(..., description="The answer from the model")
    sources: List[Source] = Field(..., description="List of sources used for the answer")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "The standard working hours are 9 AM to 5 PM, Monday through Friday.",
                "sources": [
                    {
                        "document": "employee_handbook.pdf",
                        "page": "5"
                    }
                ]
            }
        }

# Remove redundant Answer class since we have QuestionResponse