from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from langgraph.prebuilt import ToolNode

from .base import BaseAgentExecutor
from ..typing import LLMOutput, PlanExecute as AgentState
from utils.config import LOCAL_DEBUG


class AgentNodeActions(BaseAgentExecutor):

    def __init__(
        self,
        *,
        agent_llm: BaseChatModel,
        agent_tools: list[BaseTool],
    ) -> None:
        self._tool_node_ = ToolNode(agent_tools)
        # bind the tools to the LLM that will be used in the Agent Executor
        llm_with_tools = agent_llm.bind_tools(agent_tools)
        super().__init__(llm_with_tools=llm_with_tools)

    async def _run_llm_(
        self, state: AgentState, config: RunnableConfig
    ) -> dict[str, list[BaseMessage] | str]:
        output: LLMOutput = await self._runnable_llm.ainvoke(
            state,
            config=config,
            stream_mode="values"
        )
        response = output.pop("output")
        if isinstance(response, AIMessage):
            if state["is_last_step"] and response.tool_calls:
                response = AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
        # if response is not AIMessage, then there was an error.
        # if the LLM call resulted in an error, we need to connect to an
        # actual person and look into what's wrong
        if LOCAL_DEBUG:
            print(f"LLM Response: {repr(response)}\n")
        return {"messages": [response], **output}

    def _tool_control_(self, state: AgentState) -> Literal["call tools", "exit"]:  # noqa: E501
        last_message: AIMessage = state["messages"][-1]
        if last_message.tool_calls:
            return "call tools"
        else:
            return "exit"
