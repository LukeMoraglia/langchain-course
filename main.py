from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

@tool
def search(query: str) -> str:
    """
    Tool that searches the internet.
    Args:
        query: The query to search for.
    Returns:
       The search results.
    """
    print (f"Searching for {query}")
    return "Tokyo weather is sunny."

llm = ChatOpenAI(model_name="gpt-6-luna")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in Tokyo?")})
    print(result)


if __name__ == "__main__":
    main()
