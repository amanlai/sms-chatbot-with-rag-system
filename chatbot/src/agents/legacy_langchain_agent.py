from datetime import datetime

# from chainlit import AsyncLangchainCallbackHandler

from langchain_openai import ChatOpenAI

from langchain.agents import AgentExecutor, create_openai_tools_agent

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.tools import BaseTool, create_retriever_tool

from .tools import datetime_tools
from .prompt_templates import (
    LEGACY_MAIN_PROMPT_SYSTEM_MESSAGE,
    LEGACY_MAIN_PROMPT_HUMAN_MESSAGE
)
from utils.config import (
    AGENT_MODEL_NAME as model,
    AGENT_TEMPERATURE as temperature,
    DEBUG,
)


class LegacyAgent:

    def __init__(
        self,
        retriever
        # vector_store=None,
        # temperature=0.1,
        # system_message="",
        # search_type="similarity",
        # k=5,
        # **kwargs,
    ) -> None:
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.retriever = retriever
        # self.retriever = self._get_retriever(vector_store, search_type, k)
        self.tools = self._get_tools()
        # self.main_prompt = LegacyMainPromptTemplate(
        #     system_message=system_message,
        #     tool_names=[tool.name for tool in self.tools],
        # )
        self.executor = self._get_agent_executor()

    # def _get_retriever(self, vector_store, search_type, k):
    #     """
    #     Construct retriever from vector store object
    #     """
    #     retriever = vector_store.as_retriever(
    #         search_type=search_type,
    #         search_kwargs={"k": k}
    #     )
    #     return retriever

    def _get_tools(self) -> list[BaseTool]:
        """
        Creates the list of tools to be used by the Agent Executor
        """
        retriever_tool = create_retriever_tool(
            self.retriever,
            "search-document",
            "Searches and retrieves information from the document",
        )
        tools = [*datetime_tools, retriever_tool]
        return tools

    def _get_agent_executor(
        self,
        max_iterations: int = 15,
        handle_parsing_errors: bool = True,
        early_stopping_method: str = "generate",
    ) -> AgentExecutor:
        """
        Create a langchain agent executor
        """
        messages = [
            SystemMessagePromptTemplate.from_template(
                LEGACY_MAIN_PROMPT_SYSTEM_MESSAGE
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate.from_template(
                LEGACY_MAIN_PROMPT_HUMAN_MESSAGE
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        today = datetime.today()
        prompt = (
            ChatPromptTemplate
            .from_messages(messages)
            .partial(
                current_year=today.year,
                today=today,
                tool_names=[tool.name for tool in self.tools]
            )
        )
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
            # prompt=self.main_prompt.prompt,
        )
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=DEBUG,
            max_iterations=max_iterations,
            handle_parsing_errors=handle_parsing_errors,
            return_intermediate_steps=DEBUG,
            early_stopping_method=early_stopping_method,
            # callbacks=[
            #     AsyncLangchainCallbackHandler(stream_final_answer=STREAM)
            # ],
        )
        return agent_executor
