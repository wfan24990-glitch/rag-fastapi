from pydantic import BaseModel
from typing import List, Optional, Dict

class IngestRequest(BaseModel):
    text: str
    source: Optional[str] = "local"

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class SourceItem(BaseModel):
    text: str
    score: float
    source: Optional[str] = None
    id: Optional[int] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
