from typing import Literal

from langchain_core.vectorstores import VectorStore

from langgraph.checkpoint.base import BaseCheckpointSaver

from utils.config import (
    USE_LEGACY_AGENT,
)
from .plan_executor import PlanExecutor, GraphUsingO1

if USE_LEGACY_AGENT:
    from .legacy_langchain_agent import LegacyAgent


class MainAgent(PlanExecutor):

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        checkpointer: BaseCheckpointSaver | None = None,
        interrupt_before: list[str] | Literal["*"] | None = None,
        interrupt_after: list[str] | Literal["*"] | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(vector_store=vector_store)
        if USE_LEGACY_AGENT:
            self.executor = self._legacy_agent()
        else:
            self.executor = self.compile(
                checkpointer=checkpointer,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug
            )

    def _legacy_agent(self):
        legacy_agent = LegacyAgent(retriever=self._retriever)
        return legacy_agent.executor


class MainAgentUsingO1(GraphUsingO1):
    def __init__(
        self,
        *,
        vector_store: VectorStore,
        checkpointer: BaseCheckpointSaver | None = None,
        interrupt_before: list[str] | Literal["*"] | None = None,
        interrupt_after: list[str] | Literal["*"] | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(vector_store=vector_store)
        if USE_LEGACY_AGENT:
            self.executor = self._legacy_agent()
        else:
            self.executor = self.compile(
                checkpointer=checkpointer,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug
            )

    def _legacy_agent(self):
        legacy_agent = LegacyAgent(retriever=self._retriever)
        return legacy_agent.executor
