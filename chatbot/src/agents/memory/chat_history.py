# from asyncio import gather as async_gather
# from asyncio import get_running_loop
# from contextvars import copy_context
from datetime import datetime, UTC
# from functools import partial
from json import (
    dumps as json_dumps,
    loads as json_loads
)
import logging
# from typing import Callable, ParamSpec, TypeVar, cast

from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

from langchain_mongodb import MongoDBChatMessageHistory

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.mongo_client import MongoClient
from pymongo.errors import OperationFailure, WriteError

logger = logging.getLogger(__name__)


class AsyncChatHistory:
    """Chat message history that stores history in MongoDB.
    Use a re-use a single client to take advantage of MongoDB connection
    pooling.

    Based on `langchain_core.chat_history.BaseChatMessageHistory` and
    `langchain_mongodb.chat_message_histories.MongoDBChatMessageHistory`
    objects.

    Instantiate:
        .. code-block:: python

            import ChatMessageHistory

            history = ChatMessageHistory(
                client = "your-AsyncIOMotorClient instance",
                database_name = "your-database-name",
                collection_name = "your-collection-name",
                session_id = "your-session-id",
            )

    Add / retrieve messages:
        .. code-block:: python

            # Add single message
            history.add_message(message)

            # Add batch messages
            history.add_messages([message1, message2, message3, ...])

            # Add human message
            history.add_user_message(human_message)

            # Add ai message
            history.add_ai_message(ai_message)

            # Retrieve messages
            messages = history.messages
    """

    def __init__(
        self,
        client: AsyncIOMotorClient,
        session_id: str,
        database_name: str,
        collection_name: str,
        *,
        session_id_key: str = "SessionId",
        history_key: str = "History",
        create_index: bool = True,
        index_kwargs: dict | None = None,
    ) -> None:
        self.client = client
        self.session_id = session_id
        self.database_name = database_name
        self.collection_name = collection_name
        self.session_id_key = session_id_key
        self.history_key = history_key

        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

        if create_index:
            index_kwargs = index_kwargs or {}
            self.collection.create_index(self.session_id_key, **index_kwargs)

    async def aadd_messages(self, messages: list[BaseMessage]) -> None:
        """Async add a list of messages.
        Args:
            messages: A list of BaseMessage objects to store.
        """
        try:
            await self.collection.insert_many([
                {
                    self.session_id_key: self.session_id,
                    self.history_key: json_dumps(message_to_dict(message)),
                    "created_at": datetime.now(UTC)
                }
                for message in messages
            ])
        except WriteError as err:
            logger.error(err)

    async def aget_messages(self) -> list[BaseMessage]:
        """Retrieve the messages from MongoDB"""
        try:
            cursor = self.collection.find(
                {self.session_id_key: self.session_id}
            )
        except OperationFailure as error:
            logger.error(error)
        messages = []
        async for doc in cursor:
            messages.append(json_loads(doc[self.history_key]))
        return messages_from_dict(messages)

    async def aclear(self) -> None:
        """Asynchronously clear session memory from MongoDB."""
        try:
            await self.collection.delete_many(
                {self.session_id_key: self.session_id}
            )
        except WriteError as err:
            logger.error(err)

#     def add_messages(self, messages: list[BaseMessage]) -> None:
#         """Append the message to the record in MongoDB"""
#         for message in messages:
#             try:
#                 self.collection.insert_one(
#                     {
#                         self.session_id_key: self.session_id,
#                         self.history_key: json_dumps(message_to_dict(message)),  # noqa: E501
#                         "created_at": datetime.now(UTC)
#                     }
#                 )
#             except WriteError as err:
#                 logger.error(err)

#     def get_messages(self) -> list[BaseMessage]:
#         """Retrieve the messages from MongoDB"""
#         try:
#             cursor = self.collection.find(
#                 {self.session_id_key: self.session_id}
#             )
#         except OperationFailure as error:
#             logger.error(error)
#         messages = [json_loads(doc[self.history_key]) for doc in cursor]
#         return messages_from_dict(messages)

#     def clear(self) -> None:
#         """Clear session memory from MongoDB."""
#         try:
#             self.collection.delete_many(
#                 {self.session_id_key: self.session_id}
#             )
#         except WriteError as err:
#             logger.error(err)


# P = ParamSpec("P")
# T = TypeVar("T")

# # a simple version of langchain_core.runnables.config.run_in_executor
# async def run_in_executor(
#     func: Callable[P, T],
#     *args: P.args,
#     **kwargs: P.kwargs,
# ) -> T:
#     """Run a function in an executor.

#     Args:
#         func (Callable[P, Output]): The function.
#         *args (Any): The positional arguments to the function.
#         **kwargs (Any): The keyword arguments to the function.

#     Returns:
#         Output: The output of the function.

#     Raises:
#         RuntimeError: If the function raises a StopIteration.
#     """
#     def wrapper() -> T:
#         try:
#             return func(*args, **kwargs)
#         except StopIteration as exc:
#             # StopIteration can't be set on an asyncio.Future
#             # it raises a TypeError and leaves the Future pending forever
#             # so we need to convert it to a RuntimeError
#             logger.error(exc)
#             raise RuntimeError from exc
#     # Use default executor with context copied from current context
#     return await get_running_loop().run_in_executor(
#         executor=None,
#         func=cast(Callable[..., T], partial(copy_context().run, wrapper))
#     )


class ChatHistory(MongoDBChatMessageHistory):

    def __init__(
        self,
        client: MongoClient,
        session_id: str,
        database_name: str,
        collection_name: str,
        *,
        session_id_key: str = "SessionId",
        history_key: str = "history",
        create_index: bool = True,
        index_kwargs: dict | None = None,
    ) -> None:
        """Initialize with a MongoDBChatMessageHistory instance.

        Args:
            connection_string: str
                connection string to connect to MongoDB.
            session_id: str
                arbitrary key that is used to store the messages of
                 a single chat session.
            database_name: Optional[str]
                name of the database to use.
            collection_name: Optional[str]
                name of the collection to use.
            session_id_key: Optional[str]
                name of the field that stores the session id.
            history_key: Optional[str]
                name of the field that stores the chat history.
            create_index: Optional[bool]
                whether to create an index on the session id field.
            index_kwargs: Optional[Dict]
                additional keyword arguments to pass to the index creation.
        """
        self.client = client
        self.session_id = session_id
        self.database_name = database_name
        self.collection_name = collection_name
        self.session_id_key = session_id_key
        self.history_key = history_key

        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

        if create_index:
            index_kwargs = index_kwargs or {}
            self.collection.create_index(self.session_id_key, **index_kwargs)
