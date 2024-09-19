from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from src.config import Config
from src.sentiment_analyzer import SentimentAnalyzer
from src.emotion_detector import EmotionDetector
from src.knowledge_base import KnowledgeBase
from src.tool_manager import ToolManager

class LLMWrapper:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMWrapper, cls).__new__(cls)
            try:
                cls._instance.llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key=Config.OPENAI_API_KEY)
                cls._instance.sentiment_analyzer = SentimentAnalyzer()
                cls._instance.emotion_detector = EmotionDetector()
                cls._instance.knowledge_base = KnowledgeBase()
                cls._instance.tool_manager = ToolManager()
                cls._instance.chain = cls._create_chain()
            except Exception as e:
                print(f"Error initializing LLMWrapper: {e}")
                cls._instance = None
        return cls._instance

    @staticmethod
    def _create_chain():
        template = """
        User input: {user_input}
        Sentiment: {sentiment} (score: {sentiment_score})
        Emotion: {emotion} (score: {emotion_score})
        
        Relevant financial information:
        {relevant_info}
        
        You are a financial advisor. Provide a helpful and empathetic response to the user's input,
        taking into account their sentiment and emotion. Use the relevant financial information if applicable.
        If needed, you can use tools like get_stock_price or calculate_loan_interest.

        Your response:
        """
        prompt = PromptTemplate(
            input_variables=["user_input", "sentiment", "sentiment_score", "emotion", "emotion_score", "relevant_info"],
            template=template
        )
        return LLMChain(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key=Config.OPENAI_API_KEY), prompt=prompt)

    def generate_response(self, user_input):
        try:
            sentiment, sentiment_score = self.sentiment_analyzer.analyze(user_input)
            emotion = self.emotion_detector.detect(user_input)
            relevant_info = self.knowledge_base.query(user_input)
            
            response = self.chain.run(
                user_input=user_input,
                sentiment=sentiment,
                sentiment_score=f"{sentiment_score:.2f}",
                emotion=emotion['label'],
                emotion_score=f"{emotion['score']:.2f}",
                relevant_info=relevant_info
            )
            
            # Check if tool usage is needed
            if "get_stock_price" in response.lower() or "calculate_loan_interest" in response.lower():
                tool_response = self.tool_manager.run(user_input)
                response += f"\n\nAdditional information: {tool_response}"
            
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response at the moment. Is there anything else I can help you with?"