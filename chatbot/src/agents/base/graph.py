from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph


class BaseGraph:

    def __init__(self, **kwargs: Any) -> None:
        self._workflow: StateGraph | None = None
        super().__init__(**kwargs)

    def compile(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
        interrupt_before: list[str] | Literal['*'] | None = None,
        interrupt_after: list[str] | Literal['*'] | None = None,
        debug: bool = False,
    ) -> CompiledGraph:
        return self._workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
        )

    def _get_graph(self) -> None:
        raise NotImplementedError

    def _add_nodes(self) -> None:
        raise NotImplementedError

    def _add_edges(self) -> None:
        raise NotImplementedError
