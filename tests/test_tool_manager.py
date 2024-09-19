from src.tool_manager import ToolManager

tm = ToolManager()
result = tm.run("What's the current price of AAPL stock?")
print(f"Tool Manager Response: {result}")