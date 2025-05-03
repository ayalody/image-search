from pydantic import BaseModel, Field

class TextQuery(BaseModel):
    text: str = Field(..., description="Free‑text prompt")
    k: int | None = Field(None, description="How many images (optional)")
