from .classifier import ClassifierWorker
from config import BATCH_SIZE, DEVICE

MODEL_PATH = "classifier/model"

classifier_worker = ClassifierWorker(
    model_path=MODEL_PATH, batch_size=BATCH_SIZE, device=DEVICE
)
