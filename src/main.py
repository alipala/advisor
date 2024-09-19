import sys
import os

print("Starting main.py")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
print(f"Current working directory: {os.getcwd()}")

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"Added {parent_dir} to sys.path")

try:
    from src.config import Config
    print("Config imported successfully")
    print(f"OPENAI_API_KEY set: {'Yes' if Config.OPENAI_API_KEY else 'No'}")
except ImportError as e:
    print(f"Error importing Config: {e}")
    sys.exit(1)

try:
    from src.llm_wrapper import LLMWrapper
    print("LLMWrapper imported successfully")
except ImportError as e:
    print(f"Error importing LLMWrapper: {e}")
    sys.exit(1)

def main():
    try:
        llm_wrapper = LLMWrapper()
        if llm_wrapper is None:
            raise Exception("LLMWrapper failed to initialize")
        print("LLMWrapper instance created successfully")
    except Exception as e:
        print(f"Error creating LLMWrapper instance: {e}")
        print("The bot will run with limited functionality.")
        llm_wrapper = None

    print("Welcome to the Financial Advisor Bot! How can I assist you today?")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Thank you for using the Financial Advisor Bot. Have a great day!")
                break
            if llm_wrapper:
                response = llm_wrapper.generate_response(user_input)
            else:
                response = "I apologize, but I'm currently operating with limited functionality. I can still try to assist you with general questions."
            print("Advisor:", response)
        except Exception as e:
            print(f"An error occurred while processing your input: {e}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    main()