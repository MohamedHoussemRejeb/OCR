from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Literal

class ColumnSchema(BaseModel):
  name: str
  type: Literal['number','integer','date','boolean','categorical','string'] = 'string'
  nullable: bool = True
  confidence: float | None = None

class ImportPreviewRequest(BaseModel):
  sourceType: Literal['csv','excel','ocr']
  rows: List[Dict[str, Any]] = Field(default_factory=list)
  schema: Optional[List[ColumnSchema]] = None

class ImportPreviewResponse(BaseModel):
  sample: List[Dict[str, Any]]
  schema: List[ColumnSchema]
  warnings: List[str] = Field(default_factory=list)

class OcrExtractResponse(BaseModel):
  text: str
  rows: Optional[List[Dict[str, Any]]] = None
