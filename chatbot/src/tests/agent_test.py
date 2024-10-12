from langchain_core.messages import BaseMessage, HumanMessage

from langchain_openai import ChatOpenAI

from agents.chat_agent_executor import AgentExecutor
from agents.tools import datetime_tools


async def chat_agent_driver(query: str) -> dict[str, list[BaseMessage]]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = AgentExecutor(
        agent_tools=datetime_tools,
        agent_llm=llm,
        run_only_agent=True
    )
    app = agent.executor
    answer = await app.ainvoke(
        {"messages": [HumanMessage(query)]}
    )
    return answer
