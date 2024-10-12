from typing import Any, Literal

from langchain_core.runnables import RunnableConfig

from .base import BaseEdgeChains
from ...typing import (
    ClassifyQuestion,
    LLMOutput,
    PlanExecute as State,
    RouteQuestion,
)
# from utils.config import DEBUG


class EdgeActions(BaseEdgeChains):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    async def _classify_input_(
        self, state: State, config: RunnableConfig
    ) -> Literal["answer right away", "process request", "go to help desk"]:
        output_1: LLMOutput = await self._classifier_llm.ainvoke(
            state,
            config=config,
            stream_mode="value"
        )
        response = output_1.pop("output")
        if isinstance(response, ClassifyQuestion):
            if response.binary_score == "no":
                return "answer right away"
            else:
                return await self._route_query_(state=state, config=config)
        else:
            # if the LLM result is not ClassifyQuestion object,
            # that means it resulted in error.
            # we need to handle this error differently.
            # e.g. we probably need to connect to an actual
            # person to look into what's wrong.
            # for now, we just ignore it and go to immediate_answer
            return "answer right away"

    # async def _evaluate_agent_output_(
    #     self, state: State, config: RunnableConfig
    # ) -> Literal["exit", "retrieve", "replan"]:
    #     if state.get("plan"):
    #         return "replan"
    #     else:
    #         output: LLMOutput = await self._answer_grader.ainvoke(
    #             state,
    #             config=config,
    #             stream_mode="value"
    #         )
    #         # if DEBUG:
    #         #     print(f"---- ANSWER EVALUATION -----\n{repr(output)}\n")
    #         response = output["output"]
    #         if isinstance(response, GradeAnswer):
    #             if response.binary_score == "yes":
    #                 # go to exit
    #                 return "exit"
    #             else:
    #                 # go to retriever
    #                 return "retrieve"
    #         else:
    #             # if the LLM result is not a RouteQuestion object,
    #             # that means it resulted in error.
    #             # we need to handle this error differently.
    #             # e.g. we probably need to connect to an actual person
    #             # to look into what's wrong.
    #             # for now, we just ignore it and end this session
    #             return "exit"

    def _plan_execution_control_(
        self, state: State
    ) -> Literal["continue", "exit"]:
        """
        An action taken on the conditional edge.
        Returns to the agent node if response is still empty.
        Otherwise, end the loop and return the result.
        """
        if state.get("response") is None:
            # no response is generated yet so continue with more agent calls
            return "continue"
        else:
            # we're done with the entire thing, exit right away
            return "exit"

    async def _route_query_(
        self, state: State, config: RunnableConfig
    ) -> Literal["process request", "go to help desk"]:
        output: LLMOutput = await self._router.ainvoke(
            state,
            config=config,
            stream_mode="values"
        )
        response = output["output"]
        if isinstance(response, RouteQuestion):
            if response.binary_score == "yes":
                return "process request"
            else:
                return "go to help desk"
        else:
            # if the LLM result is not a RouteQuestion object,
            # that means it resulted in error.
            # we need to handle this error differently.
            # e.g. we probably need to connect to an actual
            # person to look into what's wrong.
            # for now, we just ignore it and go to immediate_answer
            return "go to help desk"
