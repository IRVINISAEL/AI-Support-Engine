from typing import Literal
from pydantic import BaseModel, Field, field_validator

Category = Literal["ACCESS", "BUG", "FEATURE", "SECURITY"]
Priority = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
Mode = Literal["traditional", "generative"]


class AnalyzeRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    mode: Mode

    @field_validator("message")
    @classmethod
    def no_vacio(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("El mensaje no puede estar vacío")
        return v


class AnalyzeResponse(BaseModel):
    category: Category
    priority: Priority
    response: str = Field(..., min_length=1)
    requiresHumanReview: bool
    mode: Mode


class ModelOutput(BaseModel):
    """Formato que debe devolver el modelo generativo."""
    category: Category
    priority: Priority
    response: str = Field(..., min_length=1, max_length=1500)