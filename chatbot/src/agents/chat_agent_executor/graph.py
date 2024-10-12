from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from langgraph.graph import StateGraph

from .actions import AgentNodeActions
from ..base import BaseGraph
from ..typing import PlanExecute as AgentState, ConfigSchema


class AgentExecutor(BaseGraph, AgentNodeActions):

    def __init__(
        self,
        *,
        agent_llm: BaseChatModel,
        agent_tools: list[BaseTool],
        run_only_agent: bool = False
    ) -> None:
        super().__init__(
            agent_llm=agent_llm,
            agent_tools=agent_tools,
        )
        if run_only_agent:
            self._workflow = StateGraph(AgentState, config_schema=ConfigSchema)
            self._get_graph()
            self.executor = self.compile()

    def _get_graph(self) -> None:
        # add nodes
        self._workflow.add_node("agent", self._run_llm_)
        self._workflow.add_node("tools", self._tool_node_)
        # add edges
        self._workflow.set_entry_point("agent")
        self._workflow.add_edge("tools", "agent")
        self._workflow.add_conditional_edges(
            "agent",
            path=self._tool_control_,
            path_map={"call tools": "tools", "exit": "__end__"}
        )
