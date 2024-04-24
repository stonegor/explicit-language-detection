import unittest
from unittest import IsolatedAsyncioTestCase
import time
import asyncio

from server.database import (
    add_corpus,
    get_corpus,
    get_unlabeled_texts,
    label,
    delete_corpus,
)


class TestDatabase(IsolatedAsyncioTestCase):
    async def test_delete_corpus(self):

        corpus = {"corpus": [{"text": "0"}]}

        expected = None

        id = (await add_corpus(corpus))["id"]

        await delete_corpus(id)

        result = await get_corpus(id)

        self.assertEqual(expected, result)

    async def test_get_unlabeled_texts(self):

        corpus = {
            "corpus": [
                {"text": "0"},
                {"text": "1"},
                {"text": "2", "result": 0},
                {"text": "3"},
                {"text": "4"},
            ]
        }

        expected = [{"id": 1, "text": "1"}, {"id": 3, "text": "3"}]

        id = (await add_corpus(corpus))["id"]

        # label(id, [2, 3], [1, 0])

        result = await get_unlabeled_texts(id, 1, 2)

        await delete_corpus(id)

        self.assertEqual(expected, result)

    async def test_label(self):

        corpus = {
            "corpus": [
                {"text": "0"},
                {"text": "1"},
                {"text": "2", "result": 0},
                {"text": "3"},
                {"text": "4"},
            ]
        }

        expected = [{"id": 3, "text": "3"}, {"id": 4, "text": "4"}]

        id = (await add_corpus(corpus))["id"]

        await label(id, [0, 1], [1, 1])

        result = await get_unlabeled_texts(id, 0, 2)

        await delete_corpus(id)

        self.assertEqual(expected, result)


if __name__ == "__main__":
    unittest.main()
