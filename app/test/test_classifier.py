import unittest
from unittest import IsolatedAsyncioTestCase
import time
from classifier.classifier import ClassifierWorker
from server.database import (
    add_corpus,
    delete_corpus,
    retrieve_result,
    get_labeled_property,
)
from config import BATCH_SIZE, DEVICE

MODEL_PATH = "classifier/model"


class TestClassifier(IsolatedAsyncioTestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        self.worker = ClassifierWorker(
            model_path=MODEL_PATH, batch_size=BATCH_SIZE, device=DEVICE
        )

        super().__init__(methodName)

    async def test_predict(self):

        texts = [
            "Убей себя немеленно!!",
            "Обычный текст, не содержащий токсичности.",
        ]

        result = list(self.worker.clf.predict(texts))

        expected = [1, 0]

        self.assertEqual(expected, result)

    async def test_get_batch(self):

        texts = [{"id": i, "text": i} for i in range(257)]
        corpus = {"corpus": texts}

        id = (await add_corpus(corpus))["id"]

        i = 0
        async for batch in self.worker.get_batch(id):
            start = i * self.worker.batch_size
            end = start + self.worker.batch_size

            self.assertEqual(batch, texts[start:end])

            i += 1

        await delete_corpus(id)

    async def test_start_prediction_loop(self):

        examples = [
            "Убей себя немеленно!!",
            "Обычный текст, не содержащий токсичности.",
        ]

        texts = [{"id": i, "text": examples[i % 2 == 0]} for i in range(257)]
        corpus = {"corpus": texts}

        id = (await add_corpus(corpus))["id"]

        await self.worker.start_prediction_loop(id)

        results = await retrieve_result(id)

        self.assertEqual(len(results), 257)

    async def test_add_corpus_to_queue(self):

        examples = [
            "Убей себя немеленно!!",
            "Обычный текст, не содержащий токсичности.",
        ]

        texts = [{"id": i, "text": examples[i % 2 == 0]} for i in range(257)]
        corpus = {"corpus": texts, "labeled": False}

        texts1 = [{"id": i, "text": examples[i % 2 == 1]} for i in range(257)]
        corpus1 = {"corpus": texts1, "labeled": False}

        id = (await add_corpus(corpus))["id"]
        id1 = (await add_corpus(corpus1))["id"]

        ids = (id, id1)

        for id in ids:
            self.worker.add_corpus_to_queue(id)

        while not self.worker.tasks.empty():

            time.sleep(1)

        self.worker.stop()

        for id in ids:
            self.assertEqual(await get_labeled_property(id), True)


if __name__ == "__main__":
    unittest.main()
