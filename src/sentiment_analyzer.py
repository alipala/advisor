from transformers import pipeline
from src.config import Config

class SentimentAnalyzer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
            try:
                cls._instance.model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            except Exception as e:
                print(f"Error loading sentiment analysis model: {e}")
                print("Falling back to a simpler sentiment analysis method.")
                cls._instance.model = None
        return cls._instance

    def analyze(self, text):
        if self.model:
            try:
                result = self.model(text)[0]
                return result['label'], result['score']
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")

        # Fallback simple sentiment analysis
        positive_words = set(['good', 'great', 'happy', 'positive', 'optimistic'])
        negative_words = set(['bad', 'terrible', 'sad', 'negative', 'worried', 'concerned'])

        words = text.lower().split()
        positive_count = sum(word in positive_words for word in words)
        negative_count = sum(word in negative_words for word in words)

        if positive_count > negative_count:
            return 'POSITIVE', 0.7
        elif negative_count > positive_count:
            return 'NEGATIVE', 0.7
        else:
            return 'NEUTRAL', 0.5