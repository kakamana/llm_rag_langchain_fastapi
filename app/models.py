from pydantic import BaseModel
from typing import List,Field

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