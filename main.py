from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# DIY tool
from tavily import TavilyClient

tavily = TavilyClient()

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
    return tavily.search(query=query)

# or use langchain-tavily integration
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for an agent's source"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for an agent's response"""
    answer: str = Field(description="The answer to the query")
    sources: List[Source] = Field(default_factory=list, description="The sources used to generate the answer")

llm = ChatOpenAI(model_name="gpt-6-luna")
tools = [TavilySearch()]  #[search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in Plano, Texas this week?")})
    print(result)


if __name__ == "__main__":
    main()
