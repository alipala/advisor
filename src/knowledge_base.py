import json
from llama_index import SimpleDirectoryReader, VectorStoreIndex, Document, ServiceContext
from llama_index.llms import OpenAI
from src.config import Config

class KnowledgeBase:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeBase, cls).__new__(cls)
            try:
                cls._instance.index = cls._create_index()
            except Exception as e:
                print(f"Error creating knowledge base index: {e}")
                cls._instance.index = None
        return cls._instance

    @staticmethod
    def _create_index():
        try:
            with open('data/financial_texts.json', 'r') as f:
                texts = json.load(f)
            documents = [Document(text=t) for t in texts]
            
            # Create a custom ServiceContext with the gpt-3.5-turbo model
            llm = OpenAI(model="gpt-3.5-turbo", api_key=Config.OPENAI_API_KEY)
            service_context = ServiceContext.from_defaults(llm=llm)
            
            return VectorStoreIndex.from_documents(documents, service_context=service_context)
        except FileNotFoundError:
            print("Warning: financial_texts.json not found. Using empty knowledge base.")
            return VectorStoreIndex.from_documents([], service_context=service_context)
        except json.JSONDecodeError:
            print("Warning: Error parsing financial_texts.json. Using empty knowledge base.")
            return VectorStoreIndex.from_documents([], service_context=service_context)
        except Exception as e:
            print(f"Unexpected error creating knowledge base: {e}")
            return None

    def query(self, text):
        if self.index is None:
            return "I'm sorry, but I don't have access to my knowledge base at the moment."
        try:
            query_engine = self.index.as_query_engine()
            response = query_engine.query(text)
            return str(response)
        except Exception as e:
            print(f"Error querying knowledge base: {e}")
            return "I'm having trouble accessing my knowledge base. How else can I assist you?"