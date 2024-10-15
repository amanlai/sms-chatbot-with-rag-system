from asyncio import create_task, Task, wait_for
from datetime import datetime, UTC
from os import getenv
from typing import Callable, Literal, overload

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    trim_messages
)

from langchain_mongodb import MongoDBAtlasVectorSearch
# from langchain_mongodb import MongoDBChatMessageHistory

from langchain_openai import OpenAIEmbeddings

from langgraph.pregel import GraphRecursionError

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from pymongo.mongo_client import MongoClient

from twilio.http.async_http_client import AsyncTwilioHttpClient
from twilio.rest import Client
from twilio.rest.api.v2010.account.message import MessageInstance

from agents.base import ChatModelWithErrorHandling
from agents.main_agent import MainAgent, MainAgentUsingO1
from agents.memory.checkpoint import AsyncMongoDBSaver
from agents.memory.chat_history import AsyncChatHistory, ChatHistory
from agents.typing import TwilioResponseMessage

from .config import (
    CHAT_HISTORY_TRIMMER_MODEL_NAME,
    CHECKPOINT_INDEX_NAME,
    DEBUG,
    EMBEDDING_MODEL_NAME,
    INDEX_NAME,
    LOCAL,
    MAX_TOKENS_AFTER_TRIMMING,
    RECURSION_LIMIT,
    TTL_INDEX_KEY,
    TTL_EXPIRE_AFTER_SECONDS,
    MAX_GRAPH_EXECUTION_TIME as MAX_EXECUTION_TIME,
    RESPONSE_AT_MAX_EXECUTION_TIME,
    RESPONSE_AT_RECURSION_ERROR,
    TWILIO_ERROR_MESSAGE,
    USE_LEGACY_AGENT,
    USE_LLAMA_INDEX,
    USE_PLAN_EXECUTE
)


if USE_LLAMA_INDEX:
    from llama_index.core import VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from agents.memory.vector_search import MongoDBAtlasVectorSearch

BUSINESS_NAME = getenv("BUSINESS_NAME")
MONGO_CHAT_HISTORY_URI = getenv("CHAT_HISTORY_MONGO_URI")
MONGO_GROUND_TRUTH_URI = getenv("MONGO_URI")
SHORT_TERM_MEMORY_URI = getenv("SHORT_TERM_MEMORY_URI")
TWILIO_ACCOUNT_SID = getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_PHONE = getenv("TWILIO_PHONE")


class MongoDBConnection:

    db_connection_kwargs = {
        "tz_aware": True,
        "maxPoolSize": None,
        "maxConnecting": 4,
        "connectTimeoutMS": 20000,
        "timeoutMS": None,
        "serverSelectionTimeoutMS": 30000,
        "waitQueueTimeoutMS": None,
    }

    def __init__(self) -> None:
        self.db_name = BUSINESS_NAME.replace(" ", "")
        self.chat_history_client = self.ainit_connection(
            MONGO_CHAT_HISTORY_URI, name="chat history"
        )
        # self.chat_history_client = self.init_connection(
        #     MONGO_CHAT_HISTORY_URI, name="chat_history"
        # )
        self.checkpoint_client = self.ainit_connection(
            SHORT_TERM_MEMORY_URI, name="checkpoint"
        )
        self.vector_store_client = self.init_connection(
            MONGO_GROUND_TRUTH_URI, name="vector store"
        )
        if "Connection Error" in (
            self.vector_store_client,
            self.chat_history_client,
            self.checkpoint_client
        ):
            raise ConnectionFailure
        self.trimmer_llm = ChatModelWithErrorHandling(
            model=CHAT_HISTORY_TRIMMER_MODEL_NAME
        )
        self.vector_store = self.get_vector_store()
        super().__init__()

    @overload
    async def aget_chat_history(
        self, *, question: Literal["Delete chat history."], session: str
    ) -> Literal["Chat history deleted."]:
        ...

    @overload
    async def aget_chat_history(
        self, *, question: str, session: str
    ) -> AsyncChatHistory:
        ...

    async def aget_chat_history(
        self, *, question: str, session: str
    ) -> Literal["Chat history deleted."] | AsyncChatHistory:
        # instantiate chat history instance
        history = AsyncChatHistory(
            client=self.chat_history_client,
            database_name=self.db_name,
            collection_name=session,
            session_id=session,
        )
        if question == "Delete chat history.":
            await history.aclear()
            return "Chat history deleted."
        else:
            return history

    def ainit_connection(
        self, uri: str, /, name: str
    ) -> AsyncIOMotorClient | Literal["Connection Error"]:
        # https://www.mongodb.com/docs/manual/administration/connection-pool-overview/#settings # noqa: E501
        try:
            if LOCAL:
                import certifi
                client = AsyncIOMotorClient(
                    uri,
                    tlsCAFile=certifi.where(),
                    **self.db_connection_kwargs
                )
            else:
                client = AsyncIOMotorClient(uri, **self.db_connection_kwargs)
            if DEBUG:
                print(f'Connected to "{name}" client.')
        except ConnectionFailure as e:
            print(f"Connection failed: {e}")
            return "Connection Error"
        return client

    @overload
    def get_chat_history(
        self, *, question: Literal["Delete chat history."], session: str
    ) -> Literal["Chat history deleted."]:
        ...

    @overload
    def get_chat_history(
        self, *, question: str, session: str
    ) -> ChatHistory:
        ...

    def get_chat_history(
        self, *, question: str, session: str
    ) -> Literal["Chat history deleted."] | ChatHistory:
        # instantiate chat history instance
        # history = MongoDBChatMessageHistory(
        #     connection_string=MONGO_CHAT_HISTORY_URI,
        #     session_id=session,
        #     database_name=self.db_name,
        #     collection_name=session
        # )
        history = ChatHistory(
            client=self.chat_history_client,
            database_name=self.db_name,
            collection_name=session,
            session_id=session,
        )
        if question == "Delete chat history.":
            history.clear()
            return "Chat history deleted."
        else:
            return history

    def get_checkpointer(
        self,
        *,
        collection_name: str | None = None
    ) -> AsyncMongoDBSaver:
        checkpointer = AsyncMongoDBSaver(
            client=self.checkpoint_client,
            database_name=self.db_name,
            collection_name=collection_name,
            ttl_index_name=CHECKPOINT_INDEX_NAME,
            ttl_index_key=TTL_INDEX_KEY,
            ttl_expire_after_seconds=TTL_EXPIRE_AFTER_SECONDS,
        )
        return checkpointer

    def get_vector_store(self) -> MongoDBAtlasVectorSearch:
        if USE_LLAMA_INDEX:
            # Instantiate the vector store
            atlas_vector_store = MongoDBAtlasVectorSearch(
                self.vector_store_client,
                db_name=self.db_name,
                collection_name=BUSINESS_NAME,
                vector_index_name=INDEX_NAME
            )
            return VectorStoreIndex.from_vector_store(
                vector_store=atlas_vector_store,
                embed_model=OpenAIEmbedding(model=EMBEDDING_MODEL_NAME),
            )
        # connect to mongodb collection
        db = self.vector_store_client[self.db_name]
        collection = db[BUSINESS_NAME]
        # get vector store
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=OpenAIEmbeddings(
                disallowed_special=(),
                model=EMBEDDING_MODEL_NAME
            ),
            index_name=INDEX_NAME,
            relevance_score_fn="cosine",
        )
        return vector_store

    def init_connection(
        self, uri: str, /, name: str = ""
    ) -> MongoClient | Literal["Connection Error"]:
        # https://www.mongodb.com/docs/manual/administration/connection-pool-overview/#settings # noqa: E501
        try:
            if LOCAL:
                import certifi
                client = MongoClient(
                    uri,
                    tlsCAFile=certifi.where(),
                    **self.db_connection_kwargs
                )
            else:
                client = MongoClient(uri, **self.db_connection_kwargs)
            if DEBUG:
                print(f'Connected to "{name}" client.')
        except ConnectionFailure as e:
            print(f"Connection failed: {e}")
            return "Connection Error"
        return client


class AnswerGenerator(MongoDBConnection):

    def __init__(self) -> None:
        super().__init__()
        if USE_PLAN_EXECUTE:
            self.agent = MainAgent(vector_store=self.vector_store)
        else:
            self.agent = MainAgentUsingO1(vector_store=self.vector_store)

    async def create_answer(self, *, question: str, session: str) -> str:
        """
        Create an answer given
        - previous chat history
        - a new question
        """
        history = await self.aget_chat_history(
            question=question, session=session
        )
        # history = self.get_chat_history(question=question, session=session)
        # this means history is just chat history deleted
        if isinstance(history, str):
            return history
        # construct the agent and generate answer
        messages = await history.aget_messages()
        # messages = history.messages
        if USE_LEGACY_AGENT:
            result = await self.agent.executor.ainvoke(
                {"input": question, "chat_history": messages},
                stream_mode="values"
            )
            answer = result["output"]
        else:
            try:
                task = self.create_executor_task(
                    question=question,
                    chat_history=messages,
                    session=session,
                )
                result = await wait_for(task, timeout=MAX_EXECUTION_TIME)
            except TimeoutError:
                # we need to aput to the _writes collection
                # even when there is an error
                result = {"response": RESPONSE_AT_MAX_EXECUTION_TIME}
            except GraphRecursionError:
                # we need to aput to the _writes collection
                # even when there is an error
                result = {"response": RESPONSE_AT_RECURSION_ERROR}
            answer = result["response"]
        # print the result to console
        if DEBUG:
            print(f"=============== APP RESULT ==============\n{result}\n")
        # add the most recent message exchange to db
        await history.aadd_messages(
            [HumanMessage(content=question), AIMessage(content=answer)]
        )
        return answer

    def create_executor_task(
        self,
        *,
        question: str,
        chat_history: list[dict],
        session: str,
        trim_history: int | Callable | None = trim_messages,
    ) -> Task:
        if isinstance(trim_history, Callable):
            trimmed_chat_history = trim_history(
                chat_history,
                max_tokens=MAX_TOKENS_AFTER_TRIMMING,
                token_counter=self.trimmer_llm,
                strategy="last",
                # include_system=True,
                start_on="human",
            )
        elif isinstance(trim_history, int):
            trimmed_chat_history = chat_history[-trim_history:]
        else:
            trimmed_chat_history = chat_history
        checkpointer = self.get_checkpointer(collection_name=session)
        task = create_task(
            self.agent
            .compile(checkpointer=checkpointer)
            .ainvoke(
                {"input": question, "chat_history": trimmed_chat_history},
                config={
                    "recursion_limit": RECURSION_LIMIT,
                    "configurable": {
                        "thread_id": f"{BUSINESS_NAME}_{session}",
                        "agent_forget_short_memory": True
                    },
                },
                stream_mode="values"
            )
        )
        return task

    # def get_agent_executor(
    #     self,
    #     *,
    #     checkpointer: AsyncMongoDBSaver,
    # ) -> Runnable:
    #     """
    #     A convenience function that copies data from collection and
    #     builds the agent executor from them and returns it
    #     """
    #     agent = MainAgent(
    #         vector_store=self.vector_store,
    #         checkpointer=checkpointer
    #     )
    #     return agent.executor

    async def send_sms(
        self, *, input_message: str, to: str
    ) -> TwilioResponseMessage:
        """Send SMS text message and return the message status, SID, created
        date and message body back as a Pydantic BaseModel"""
        answer = await self.create_answer(question=input_message, session=to)
        http_client = AsyncTwilioHttpClient()
        client = Client(
            username=TWILIO_ACCOUNT_SID,
            password=TWILIO_AUTH_TOKEN,
            http_client=http_client,
        )
        try:
            message: MessageInstance = await client.messages.create_async(
                body=answer,
                from_=TWILIO_FROM_PHONE,
                to=to,
            )
            response = TwilioResponseMessage(
                status=message.status,
                sid=message.sid,
                date_created=message.date_created,
                body=message.body
            )
        except Exception as error:
            response = TwilioResponseMessage(
                status=type(error).__name__,
                sid=None,
                date_created=datetime.now(UTC),
                body=TWILIO_ERROR_MESSAGE.format(str(error))
            )
            # aput to writes collection
        finally:
            await http_client.close()
        return response
