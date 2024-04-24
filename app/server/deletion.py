import asyncio
from datetime import datetime, timedelta
import threading


class DeletionWorker:
    def __init__(self, collection, delta, interval):
        self.collection = collection
        self.interval = interval
        self.delta = delta
        self._stop_event = threading.Event()

    async def delete_labeled_datasets(self):
        while not self._stop_event.is_set():
            try:
                threshold_time = datetime.now() - timedelta(seconds=self.delta)
                result = await self.collection.delete_many(
                    {"labeled_at": {"$lt": threshold_time}}
                )
                print(
                    f"Deleted {result.deleted_count} datasets labeled before {threshold_time}"
                )
                await asyncio.sleep(self.interval)
            except Exception as e:
                print(f"An error occurred: {e}")
                self.stop()

    def run(self):
        thread = threading.Thread(target=self.start_loop)
        thread.daemon = True
        thread.start()

    def start_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.delete_labeled_datasets())
        loop.close()

    def stop(self):
        self._stop_event.set()
