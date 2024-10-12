from functools import partial
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
# from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableParallel
)
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_openai import ChatOpenAI
# from langchain_openai.chat_models.base import _is_pydantic_class

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from pydantic import BaseModel

from ..typing import LLMOutput

from utils.config import (
    API_ERROR_MESSAGE,
    TIMEOUT_ERROR_MESSAGE,
    OTHER_ERROR_MESSAGE,
)


class ChatModelWithErrorHandling(ChatOpenAI):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any
    ) -> LLMOutput:
        try:
            return {
                "output": await super().ainvoke(
                    input=input,
                    config=config,
                    stop=stop,
                    **kwargs
                ),
                "error": None
            }
        except APITimeoutError as terr:
            # we need to aput to the _writes collection
            # even when there is an error
            return {"output": AIMessage(TIMEOUT_ERROR_MESSAGE), "error": str(terr)}  # noqa: E501
        except (
            APIConnectionError,
            APIError,
            AuthenticationError,
            BadRequestError,
            RateLimitError
        ) as err:
            # we need to aput to the _writes collection
            # even when there is an error
            return {"output": AIMessage(API_ERROR_MESSAGE), "error": str(err)}  # noqa: E501
        except Exception as e:
            # we need to aput to the _writes collection
            # even when there is an error
            return {"output": AIMessage(OTHER_ERROR_MESSAGE), "error": str(e)}  # noqa: E501


class BaseChains:

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _get_prompt(
        self,
        *,
        human_prompt: str,
        system_prompt: str | None = None,
        partial_values: dict[str, Any] = {}
    ) -> ChatPromptTemplate:
        if system_prompt:
            messages = [
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder("chat_history", optional=True),
            ]
        else:
            messages = []
        messages.append(HumanMessagePromptTemplate.from_template(human_prompt))
        prompt = ChatPromptTemplate.from_messages(messages)
        if partial_values:
            prompt = prompt.partial(**partial_values)
        return prompt

    def _get_o1_chain(
        self,
        *,
        llm: ChatModelWithErrorHandling,
        human_prompt: str,
        partial_values: dict[str, Any] = {}
    ) -> Runnable | None:
        prompt = self._get_prompt(
            human_prompt=human_prompt,
            partial_values=partial_values
        )
        chain = prompt | llm | RunnableParallel(
            output=lambda di: di["output"].content,
            error=lambda di: di["error"]
        )
        return chain

    def _get_str_output_chain(
        self,
        *,
        llm: ChatModelWithErrorHandling,
        system_prompt: str | None = None,
        human_prompt: str | None = None,
        partial_values: dict[str, Any] = {}
    ) -> Runnable:
        prompt = self._get_prompt(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            partial_values=partial_values
        )
        chain = prompt | llm | RunnableParallel(
            output=lambda di: di["output"].content,
            error=lambda di: di["error"]
        )
        return chain

    def _get_structured_output_chain(
        self,
        *,
        llm: ChatModelWithErrorHandling,
        system_prompt: str,
        human_prompt: str,
        schema: type[BaseModel],
        strict: bool | None = None,
        partial_values: dict[str, Any] = {}
    ) -> Runnable:
        prompt = self._get_prompt(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            partial_values=partial_values
        )
        tool_name = convert_to_openai_tool(schema)["function"]["name"]
        llm = llm.bind_tools(
            [schema],
            tool_choice=tool_name,
            parallel_tool_calls=False,
            strict=strict
        )
        parser = RunnableParallel(
            output=partial(self._parser_ainvoke, schema=schema),
            error=lambda di: di["error"]
        )
        return prompt | llm | parser

    async def _parser_ainvoke(self, di: LLMOutput, schema: type[BaseModel]) -> BaseModel:  # noqa: E501
        output_parser = PydanticToolsParser(
            tools=[schema],
            first_tool_only=True,
        )
        return await output_parser.ainvoke(di["output"])
