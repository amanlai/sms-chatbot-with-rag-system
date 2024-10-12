from datetime import datetime

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.runnables import Runnable

from ..prompt_templates import RAG_SYSTEM_PROMPT

from utils.config import TIMEZONE


class BaseAgentExecutor:

    def __init__(
        self,
        *,
        llm_with_tools: Runnable,
    ) -> None:
        self._system_message = self._define_system_message()
        self._runnable_llm = self._get_chain(llm_with_tools=llm_with_tools)

    def _define_system_message(
        self
    ) -> SystemMessage | ChatPromptTemplate | None:
        # prompt = SystemMessage("You are a helpful assistant")
        # msg = "You are a helpful assistant."
        prompt = RAG_SYSTEM_PROMPT
        today = datetime.now(TIMEZONE)
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(prompt),
                MessagesPlaceholder("messages")
            ]
        ).partial(
            current_year=today.year,
            today=today.isoformat(),
            now=today,
        )
        # prompt = None
        return prompt

    def _get_chain(
        self,
        llm_with_tools: Runnable,
    ) -> Runnable:
        if isinstance(self._system_message, SystemMessage):
            runnable_llm = self._message_modifier | llm_with_tools
        elif isinstance(self._system_message, ChatPromptTemplate):
            runnable_llm = self._system_message | llm_with_tools
        elif self._system_message is None:
            runnable_llm = llm_with_tools
        else:
            raise ValueError(
                "Got unexpected type for `system_message`: "
                f"{type(self._system_message)}"
                "Expected a SystemMessage or a ChatPromptTemplate."
            )
        return runnable_llm

    def _message_modifier(self, messages: list[BaseMessage]) -> list[BaseMessage]:  # noqa: E501
        return [self._system_message, *messages]
