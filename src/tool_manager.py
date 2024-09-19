from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from src.config import Config

class ToolManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolManager, cls).__new__(cls)
            cls._instance.agent = cls._create_agent()
        return cls._instance

    @staticmethod
    def _create_agent():
        llm = OpenAI(temperature=0, api_key=Config.OPENAI_API_KEY)
        tools = [
            Tool(
                name="Stock Price",
                func=lambda x: f"The current stock price of {x} is $100.00",
                description="Get the current price of a stock. Input should be the stock symbol."
            ),
            Tool(
                name="Loan Interest",
                func=lambda x: f"The interest on a loan of ${x.split(',')[0]} at {float(x.split(',')[1])*100}% for {x.split(',')[2]} years is ${float(x.split(',')[0]) * float(x.split(',')[1]) * float(x.split(',')[2]):.2f}",
                description="Calculate loan interest. Input should be in the format: principal,rate,time"
            )
        ]
        return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    def run(self, query):
        return self.agent.run(query)