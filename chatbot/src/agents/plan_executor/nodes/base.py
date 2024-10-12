from datetime import datetime
from functools import partial
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from ...base import BaseChains, ChatModelWithErrorHandling
from ...prompt_templates import (
    IMMEDIATE_ANSWERER_HUMAN_PROMPT,
    IMMEDIATE_ANSWERER_SYSTEM_PROMPT,
    O1_MODEL_PROMPT,
    PLANNER_HUMAN_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    QUERY_PROCESSOR_HUMAN_PROMPT,
    QUERY_PROCESSOR_SYSTEM_PROMPT,
    REPLANNER_HUMAN_PROMPT,
    REPLANNER_SYSTEM_PROMPT
)
from ...typing import (
    Plan,
    Act,
    RetrieverInput
)
from utils.config import (
    TIMEZONE,
    RETRIEVER_POST_FILTER_MIN_SIMILARITY_SCORE,
    USE_PLAN_EXECUTE
)


class BaseNodeChains(BaseChains):

    def __init__(
        self,
        *,
        agent_llm: ChatModelWithErrorHandling,
        vector_store: VectorStore,
        node_action_llm: ChatModelWithErrorHandling,
        planner_llm: ChatModelWithErrorHandling | None = None,
        **kwargs: Any
    ) -> None:
        self._retrieve_prompt = PromptTemplate.from_template("{content}")
        self._query_processor = self._get_query_processor(llm=node_action_llm)
        self._answerer = self._get_answerer(llm=node_action_llm)
        self._retriever = self._get_retriever(vector_store=vector_store)
        # agent_tools += [self._retriever_tool(retriever)]
        if USE_PLAN_EXECUTE:
            self._planner = self._get_planner(llm=planner_llm)
            self._replanner = self._get_replanner(llm=planner_llm)
            super().__init__(agent_llm=agent_llm, **kwargs)
        else:
            self._agent_llm = self._get_o1_agent(llm=agent_llm)
            super().__init__(**kwargs)

    async def _format_docs(self, docs: list[Document]) -> str:
        return "\n\n".join(
            [
                await self._retrieve_prompt.aformat(content=doc.page_content)
                for doc in docs
            ]
        )

    # def _format_template(self, input: str, context: str) -> str:
    #     return RAG_HUMAN_PROMPT.format(input=input, context=context)

    def _get_answerer(self, llm: ChatModelWithErrorHandling) -> Runnable:
        chain = self._get_str_output_chain(
            llm=llm,
            system_prompt=IMMEDIATE_ANSWERER_SYSTEM_PROMPT,
            human_prompt=IMMEDIATE_ANSWERER_HUMAN_PROMPT,
        )
        return chain

    def _get_o1_agent(self, llm: ChatModelWithErrorHandling) -> Runnable:
        today = datetime.now(TIMEZONE)
        chain = self._get_o1_chain(
            llm=llm,
            human_prompt=O1_MODEL_PROMPT,
            partial_values={
                "current_year": today.year,
                "today": today.isoformat()
            }
        )
        return chain

    def _get_planner(self, llm: ChatModelWithErrorHandling | None) -> Runnable:
        if llm:
            chain = self._get_structured_output_chain(
                llm=llm,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                human_prompt=PLANNER_HUMAN_PROMPT,
                schema=Plan
            )
            return chain
        else:
            return None

    def _get_query_processor(self, llm: ChatModelWithErrorHandling) -> Runnable:  # noqa: E501
        chain = self._get_str_output_chain(
            llm=llm,
            system_prompt=QUERY_PROCESSOR_SYSTEM_PROMPT,
            human_prompt=QUERY_PROCESSOR_HUMAN_PROMPT,
        )
        return chain

    def _get_replanner(self, llm: ChatModelWithErrorHandling | None) -> Runnable:  # noqa: E501
        if llm:
            chain = self._get_structured_output_chain(
                llm=llm,
                system_prompt=REPLANNER_SYSTEM_PROMPT,
                human_prompt=REPLANNER_HUMAN_PROMPT,
                schema=Act
            )
            return chain
        else:
            return None

    def _get_retriever(
        self,
        vector_store: VectorStore,
        *,
        search_type: str = "similarity",
        k: int = 3,
        include_scores: bool = True,
    ) -> VectorStoreRetriever:
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={
                "include_scores": include_scores,
                "k": k,
                "post_filter_pipeline": [
                    {
                        "$match": {
                            "score": {"$gt": RETRIEVER_POST_FILTER_MIN_SIMILARITY_SCORE}  # noqa: E501
                        }
                    }
                ]
            }
        )
        return retriever

    async def _get_relevant_docs(
        self,
        input: str,
        *,
        retriever: BaseRetriever,
    ) -> str:
        docs = await retriever.ainvoke(input)
        return await self._format_docs(docs)

    def _retriever_tool(
        self,
        retriever: BaseRetriever,
    ) -> Tool:
        coro = partial(
            self._get_relevant_docs,
            retriever=retriever,
        )
        return Tool(
            name="search-document",
            description=(
                "Searches and retrieves information from the document"
            ),
            func=None,
            coroutine=coro,
            args_schema=RetrieverInput,
        )
