from fastapi import APIRouter, Body, HTTPException
from fastapi.encoders import jsonable_encoder
from bson.objectid import ObjectId
from fastapi.responses import JSONResponse

from server.models.corpus import CorpusSchema, TextSchema
from typing import List
from server.database import (
    retrieve_result,
    add_corpus,
    get_labeled_property,
)

from classifier.inference import classifier_worker


router = APIRouter()


@router.post(
    "/",
    responses={
        202: {
            "content": {
                "application/json": {"example": {"id": "661ff9400fcf8c9cce8c0990"}}
            }
        },
    },
)
async def add_request(
    texts: List[TextSchema] = Body(
        ...,
        example=[
            {
                "text": "Sample text for sentiment analysis",
            },
            {
                "text": "Another text for sentiment analysis",
            },
        ],
    )
):
    corpus = CorpusSchema(corpus=texts)
    encoded_corpus = jsonable_encoder(corpus)
    id = await add_corpus(encoded_corpus)

    classifier_worker.add_corpus_to_queue(id["id"])

    return JSONResponse(status_code=202, content=id)


@router.get(
    "/{id}",
    responses={
        404: {"description": "Corpus not found"},
        400: {"description": "Invalid corpus id"},
        202: {"description": "Corpus is being processed"},
        200: {
            "content": {
                "application/json": {
                    "example": [
                        {
                            "text": "Sample text for sentiment analysis",
                            "result": 0,
                        },
                        {
                            "text": "Another sample text for sentiment analysis",
                            "result": 0,
                        },
                    ],
                }
            }
        },
    },
)
async def get_result(id):
    try:
        id = ObjectId(id)
    except:
        raise HTTPException(status_code=400, detail="Invalid corpus id")

    labeled = await get_labeled_property(id)

    if labeled == None:
        raise HTTPException(status_code=404, detail="Corpus not found")

    labeled_corpus = await retrieve_result(id)

    if not labeled:
        return JSONResponse(status_code=202, content=labeled_corpus)

    return JSONResponse(status_code=200, content=labeled_corpus)
