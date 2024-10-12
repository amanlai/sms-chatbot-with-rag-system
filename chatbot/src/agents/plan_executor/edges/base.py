from typing import Any

from langchain_core.runnables import Runnable

from ...base import BaseChains, ChatModelWithErrorHandling
from ...typing import RouteQuestion, ClassifyQuestion
from ...prompt_templates import (
    # IMMEDIATE_ANSWER_EVALUATOR_HUMAN_PROMPT,
    # IMMEDIATE_ANSWER_EVALUATOR_SYSTEM_PROMPT,
    HOTEL_QUESTION_ROUTER_HUMAN_PROMPT,
    HOTEL_QUESTION_ROUTER_SYSTEM_PROMPT,
    QUESTION_RELEVANCY_CLASSIFIER_HUMAN_PROMPT,
    QUESTION_RELEVANCY_CLASSIFIER_SYSTEM_PROMPT,
)


class BaseEdgeChains(BaseChains):

    def __init__(
        self,
        *,
        evaluator_llm: ChatModelWithErrorHandling,
        **kwargs: Any
    ) -> None:
        self._classifier_llm = self._get_classifier(llm=evaluator_llm)
        self._router = self._get_router(llm=evaluator_llm)
        # self._answer_grader = self._get_evaluator(llm=evaluator_llm)
        super().__init__(**kwargs)

    # def _get_evaluator(self, llm: ChatModelWithErrorHandling) -> Runnable:
    #     chain = self._get_structured_output_chain(
    #         llm=llm,
    #         system_prompt=IMMEDIATE_ANSWER_EVALUATOR_SYSTEM_PROMPT,
    #         human_prompt=IMMEDIATE_ANSWER_EVALUATOR_HUMAN_PROMPT,
    #         schema=GradeAnswer,
    #     )
    #     return chain

    def _get_classifier(self, llm: ChatModelWithErrorHandling) -> Runnable:
        chain = self._get_structured_output_chain(
            llm=llm,
            system_prompt=QUESTION_RELEVANCY_CLASSIFIER_SYSTEM_PROMPT,
            human_prompt=QUESTION_RELEVANCY_CLASSIFIER_HUMAN_PROMPT,
            schema=ClassifyQuestion,
        )
        return chain

    def _get_router(self, llm: ChatModelWithErrorHandling) -> Runnable:
        chain = self._get_structured_output_chain(
            llm=llm,
            system_prompt=HOTEL_QUESTION_ROUTER_SYSTEM_PROMPT,
            human_prompt=HOTEL_QUESTION_ROUTER_HUMAN_PROMPT,
            schema=RouteQuestion,
        )
        return chain
