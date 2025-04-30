from pydantic import BaseModel, Field

class TextQuery(BaseModel):
    text: str = Field(..., description="Free-form search prompt")
    k:   int | None = Field(None, ge=1, le=50,
                            description="Number of images to return "
                                        "(falls back to default if omitted)")
