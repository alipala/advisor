from transformers import pipeline
from src.config import Config

class EmotionDetector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmotionDetector, cls).__new__(cls)
            cls._instance.model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)
        return cls._instance

    def detect(self, text):
        results = self.model(text)[0]
        return max(results, key=lambda x: x['score'])