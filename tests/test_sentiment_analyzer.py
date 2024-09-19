from src.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze("I'm really excited about my new investment strategy!")
print(f"Sentiment: {result[0]}, Score: {result[1]}")