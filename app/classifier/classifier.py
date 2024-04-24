from .model import MeanPoolingElectraForSequenceClassification
from transformers import AutoTokenizer
from typing import List, Dict
import numpy as np
from server.database import get_unlabeled_texts, label, label_corpus
import asyncio
import threading
from queue import Queue, Empty
import torch


class Classifier:
    def __init__(self, model_path, device=None) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = MeanPoolingElectraForSequenceClassification.from_pretrained(
            model_path
        ).to(self.device)

    def predict(self, texts: List[str]) -> List[int]:
        texts = self.preprocess(texts)
        tokenized_text = self.tokenize(texts)

        tokenized_text = {
            key: value.to(self.device) for key, value in tokenized_text.items()
        }

        with torch.no_grad():
            logits = self.model(**tokenized_text)[0]

        labels = logits.argmax(axis=1)

        return labels.cpu().numpy().tolist()

    def tokenize(self, texts: List[str]) -> Dict:
        return self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

    def preprocess(self, texts: List[str]) -> List[str]:
        texts = list(map(lambda x: x.strip(), texts))
        texts = list(map(lambda x: bytes(x, "utf-8").decode("utf-8", "ignore"), texts))

        return texts


class ClassifierWorker:
    def __init__(self, model_path, batch_size, device, interval=1):
        self.tasks = Queue()
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run)
        self.loop = asyncio.new_event_loop()
        self.interval = interval
        self.clf = Classifier(model_path, device=device)
        self.batch_size = batch_size
        self.device = device
        self.thread.daemon = True
        self.thread.start()

    def add_task(self, coro):
        self.tasks.put(coro)

    def run(self):
        asyncio.set_event_loop(self.loop)
        while not self._stop_event.is_set():
            try:
                task = self.tasks.get(timeout=self.interval)
            except Empty:
                continue
            if task is None:  # Sentinel value to indicate shutdown
                break
            coroutine = asyncio.iscoroutine(task)
            if coroutine:
                self.loop.run_until_complete(task)
            else:
                task()

    def stop(self):
        self.add_task(None)  # Add sentinel value to the queue
        self._stop_event.set()
        self.thread.join()
        self.loop.stop()
        self.loop.close()

    def add_corpus_to_queue(self, id: str):
        self.add_task(self.start_prediction_loop(id))

    async def start_prediction_loop(self, id: str) -> None:
        async for batch in self.get_batch(id):
            indeces = [x["id"] for x in batch]
            texts = [x["text"] for x in batch]

            labels = self.clf.predict(texts)

            await label(id, indeces, labels)

        await label_corpus(id)

    async def get_batch(self, id: str, start_ind: int = 0):

        batch = await get_unlabeled_texts(id, start_ind, self.batch_size)

        num_batches = 0

        yield batch

        while len(batch) == self.batch_size:
            num_batches += 1
            batch = await get_unlabeled_texts(
                id, num_batches * self.batch_size, self.batch_size
            )
            if len(batch) == 0:
                break
            yield batch
