from typing import List, Optional
from pydantic import BaseModel, Field


class TextSchema(BaseModel):

    text: str
    result: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "text": "Sample text for sentiment analysis",
                "result": None,
            }
        }


class CorpusSchema(BaseModel):

    corpus: List[TextSchema]
    labeled: bool = Field(default=False)

    class Config:
        schema_extra = {
            "example": {
                "corpus": [
                    {
                        "text": "Sample text for sentiment analysis",
                    },
                    {
                        "text": "Another text for sentiment analysis",
                    },
                ],
            }
        }
