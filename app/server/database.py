import asyncio
import motor.motor_asyncio
from bson.objectid import ObjectId
from config import MONGO_DETAILS, EXPIRATION_TIME, DELETION_POOLING_INTERVAL
from typing import Dict, List
from datetime import datetime
from .deletion import DeletionWorker

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
client.get_io_loop = asyncio.get_running_loop

database = client.corpus
corpus_collection = database.get_collection("corpus_collection")

worker = DeletionWorker(
    corpus_collection, delta=EXPIRATION_TIME, interval=DELETION_POOLING_INTERVAL
)
worker.run()


def text_helper(text) -> dict:
    return {
        "text": text["text"],
        "result": text["result"],
    }


def corpus_helper(corpus) -> dict:
    return {
        "id": str(corpus["_id"]),
        "corpus": [text_helper(text) for text in corpus["corpus"]],
    }


def id_helper(corpus) -> dict:
    return {"id": str(corpus["_id"])}


async def add_corpus(corpus_data: dict) -> dict:
    corpus = await corpus_collection.insert_one(corpus_data)
    new_corpus = await corpus_collection.find_one({"_id": corpus.inserted_id})
    return id_helper(new_corpus)


async def get_labeled_property(id: str) -> dict:
    corpus = await corpus_collection.find_one({"_id": ObjectId(id)}, {"labeled": 1})
    if corpus:
        return corpus["labeled"]

    return None


async def retrieve_result(id: str) -> List[Dict]:
    """
    Returns:
    - list: A list of dictionaries [{"id": <id>, "text": <text>, "result": <result>}]
    """
    pipeline = [
        {"$match": {"_id": ObjectId(id)}},
        {"$unwind": "$corpus"},
        {"$match": {"corpus.result": {"$in": [0, 1]}}},
        {
            "$project": {
                "_id": 0,
                "id": "$corpus.id",
                "text": "$corpus.text",
                "result": "$corpus.result",
            }
        },
    ]

    cursor = corpus_collection.aggregate(pipeline)

    texts_list = await cursor.to_list(None)

    return texts_list


async def get_corpus(id: str) -> dict:
    corpus = await corpus_collection.find_one({"_id": ObjectId(id)})
    if corpus:
        return corpus_helper(corpus)


async def label(id: str, indices: List[int], labels: List[bool]) -> None:

    update_operations = {}

    for index, label in zip(indices, labels):
        update_operations[f"corpus.{index}.result"] = label

    corpus_collection.update_one({"_id": ObjectId(id)}, {"$set": update_operations})


async def delete_corpus(id: str) -> None:

    corpus_collection.delete_one({"_id": ObjectId(id)})


async def get_unlabeled_texts(id: str, start: int, count: int) -> List[Dict]:
    """
    Returns:
    - list: A list of dictionaries [{"id": <id>, "text": <text>}]
    """
    pipeline = [
        {"$match": {"_id": ObjectId(id)}},
        {"$unwind": {"path": "$corpus", "includeArrayIndex": "index"}},
        {"$skip": start},
        {"$match": {"corpus.result": {"$nin": [0, 1]}}},
        {
            "$project": {
                "_id": 0,
                "id": "$index",
                "text": "$corpus.text",
                "result": "$corpus.result",
            }
        },
        {"$limit": count},
    ]

    cursor = corpus_collection.aggregate(pipeline)

    texts_list = await cursor.to_list(length=count)

    return texts_list


async def label_corpus(id: str, labeled=True):
    corpus_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"labeled": labeled, "labeled_at": datetime.now()}},
    )
