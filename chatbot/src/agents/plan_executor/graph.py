from langchain_core.vectorstores import VectorStore

from langgraph.graph import StateGraph

from .edges.actions import EdgeActions
from .nodes.actions import NodeActions
from ..base import BaseGraph, ChatModelWithErrorHandling
from ..chat_agent_executor import AgentExecutor
from ..tools import datetime_tools
from ..typing import BaseState, ConfigSchema, PlanExecute

from utils.config import (
    AGENT_MODEL_NAME,
    AGENT_TEMPERATURE,
    EVALUATOR_MODEL_NAME,
    EVALUATOR_TEMPERATURE,
    O1_MODEL_NAME,
    O1_MODEL_TEMPERATURE,
    PLANNER_MODEL_NAME,
    PLANNER_TEMPERATURE,
    NODE_ACTION_MODEL_NAME,
    NODE_ACTION_TEMPERATURE,
    OPENAI_CLIENT_TIMEOUT as TIMEOUT,
    MAX_RETRIES,
)


class PlanExecutorGraph(
    NodeActions,
    EdgeActions,
    AgentExecutor,  # RAG Agent
    BaseGraph
):

    def __init__(
        self,
        *,
        vector_store: VectorStore,
    ) -> None:
        """
        checkpointer, interrupt_before, interrupt_after, debug kwargs
        are passed to AgentExecutor's compile() method.
        For the same kwargs of the plan executor object, pass to its
        compile() method.
        """
        agent_llm = ChatModelWithErrorHandling(
            model=AGENT_MODEL_NAME,
            temperature=AGENT_TEMPERATURE,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        evaluator_llm = ChatModelWithErrorHandling(
            model=EVALUATOR_MODEL_NAME,
            temperature=EVALUATOR_TEMPERATURE,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        node_action_llm = ChatModelWithErrorHandling(
            model=NODE_ACTION_MODEL_NAME,
            temperature=NODE_ACTION_TEMPERATURE,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        planner_llm = ChatModelWithErrorHandling(
            model=PLANNER_MODEL_NAME,
            temperature=PLANNER_TEMPERATURE,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        super().__init__(
            agent_llm=agent_llm,
            planner_llm=planner_llm,
            agent_tools=datetime_tools,
            vector_store=vector_store,
            node_action_llm=node_action_llm,
            evaluator_llm=evaluator_llm,
        )
        self._workflow = StateGraph(PlanExecute, config_schema=ConfigSchema)
        self._get_graph()

    def _add_edges(self) -> None:
        self._workflow.set_conditional_entry_point(
            path=self._classify_input_,
            path_map={
                "answer right away": "immediate answer",
                "process request": "request",
                "go to help desk": "process query"
            }
        )
        # self._workflow.add_edge("__start__", "process query")
        # self._workflow.add_conditional_edges(
        #     "process query",
        #     path=self._route_query_,
        #     path_map={
        #         "process request": "request",
        #         "go to help desk": "agent entry"
        #     }
        # )
        # self._workflow.add_conditional_edges(
        #     # "generate_answer",
        #     "agent exit",
        #     path=self._evaluate_agent_output_,
        #     path_map={
        #         "exit": "__end__",
        #         "retrieve": "retriever",
        #         "replan": "replanner",
        #     }
        # )
        self._workflow.add_edge("process query", "retriever")
        self._workflow.add_edge("retriever", "planner")
        self._workflow.add_edge("planner", "agent entry")
        self._workflow.add_edge("agent entry", "agent")
        # replace the tools control with the new one that directs to replanner
        self._workflow.add_conditional_edges(
            "agent",
            path=self._tool_control_,
            path_map={"call tools": "tools", "exit": "agent exit"}
        )
        self._workflow.add_edge("agent exit", "replanner")
        self._workflow.add_conditional_edges(
            "replanner",
            path=self._plan_execution_control_,
            path_map={"exit": "__end__", "continue": "manage memory"}
        )
        self._workflow.add_edge("manage memory", "agent entry")
        self._workflow.add_edge("request", "__end__")
        self._workflow.add_edge("immediate answer", "__end__")

    def _add_nodes(self) -> None:
        self._workflow.add_node("process query", self._process_query_)
        self._workflow.add_node("request", self._fulfill_request_)
        self._workflow.add_node("immediate answer", self._generate_immediate_answer_)  # noqa: E501
        self._workflow.add_node("retriever", self._retriever_)
        self._workflow.add_node("planner", self._planner_)
        self._workflow.add_node("agent entry", self._agent_entry_)
        self._workflow.add_node("agent exit", self._agent_exit_)
        self._workflow.add_node("replanner", self._replanner_)
        self._workflow.add_node("manage memory", self._clear_intermediate_steps_from_previous_agent_run_)  # noqa: E501

    def _add_subgraph_from_agent(self) -> None:
        # get the chat agent from the AgentExecutor
        super()._get_graph()
        # remove the chat agent's starting point
        self._workflow.edges.discard(("__start__", "agent"))
        # remove the tools control flow conditional edges
        self._workflow.branches.pop("agent")

    def _get_graph(self) -> None:
        self._add_subgraph_from_agent()
        self._add_edges()
        self._add_nodes()


class Graph(NodeActions, EdgeActions, BaseGraph):

    def __init__(
        self,
        *,
        vector_store: VectorStore,
    ) -> None:
        """
        checkpointer, interrupt_before, interrupt_after, debug kwargs
        are passed to AgentExecutor's compile() method.
        For the same kwargs of the plan executor object, pass to its
        compile() method.
        """
        agent_llm = ChatModelWithErrorHandling(
            model=O1_MODEL_NAME,
            temperature=O1_MODEL_TEMPERATURE,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        evaluator_llm = ChatModelWithErrorHandling(
            model=EVALUATOR_MODEL_NAME,
            temperature=EVALUATOR_TEMPERATURE,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        node_action_llm = ChatModelWithErrorHandling(
            model=NODE_ACTION_MODEL_NAME,
            temperature=NODE_ACTION_TEMPERATURE,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        super().__init__(
            agent_llm=agent_llm,
            vector_store=vector_store,
            node_action_llm=node_action_llm,
            evaluator_llm=evaluator_llm,
        )
        self._workflow = StateGraph(BaseState, config_schema=ConfigSchema)
        self._get_graph()

    def _add_edges(self) -> None:
        self._workflow.set_conditional_entry_point(
            path=self._route_query_,
            path_map={
                "process request": "request",
                "go to help desk": "process query"
            }
        )
        self._workflow.add_edge("process query", "retriever")
        self._workflow.add_edge("retriever", "agent")
        self._workflow.add_edge("agent", "__end__")
        self._workflow.add_edge("request", "__end__")

    def _add_nodes(self) -> None:
        self._workflow.add_node("process query", self._process_query_)
        self._workflow.add_node("agent", self._call_o1_)
        self._workflow.add_node("request", self._fulfill_request_)
        self._workflow.add_node("retriever", self._retriever_)

    def _get_graph(self) -> None:
        self._add_edges()
        self._add_nodes()
