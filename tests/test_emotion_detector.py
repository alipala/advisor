from src.emotion_detector import EmotionDetector

detector = EmotionDetector()
result = detector.detect("I'm worried about the stock market crash.")
print(f"Emotion: {result['label']}, Score: {result['score']}")