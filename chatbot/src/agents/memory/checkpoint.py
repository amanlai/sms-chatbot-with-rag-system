# adapted from langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver and
# https://langchain-ai.github.io/langgraph/how-tos/persistence_mongodb/

from asyncio import run as async_run
# from contextlib import asynccontextmanager
from datetime import datetime, UTC
from types import TracebackType
from typing import Any, AsyncIterator, Sequence, Self

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import OperationFailure
from pymongo.operations import UpdateOne


class AsyncMongoDBSaver(BaseCheckpointSaver):
    """A checkpoint saver that stores checkpoints in a MongoDB database
    asynchronously.
    """

    client: AsyncIOMotorClient
    db: AsyncIOMotorDatabase
    collection_name: str

    def __init__(
        self,
        *,
        client: AsyncIOMotorClient,
        database_name: str,
        collection_name: str,
        ttl_index_name: str | None = None,
        ttl_index_key: str | None = None,
        ttl_expire_after_seconds: float = 60
    ) -> None:
        super().__init__()
        self.client = client
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.write_collection = self.db[f"{collection_name}_writes"]
        self.ttl_index_key = ttl_index_key
        async_run(
            self.create_index(
                name=ttl_index_name,
                expire_after_seconds=ttl_expire_after_seconds
            )
        )

    async def __enter__(self) -> Self:
        return self

    async def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        return True

    # @classmethod
    # @asynccontextmanager
    # async def with_context(
    #     cls,
    #     host: str,
    #     *,
    #     port: int | None = None,
    #     database_name: str | None = None,
    #     collection_name: str | None = None,
    #     ttl_index_name: str | None = None,
    #     ttl_index_key: str | None = None,
    #     ttl_expire_after_seconds: float = 60,
    #     is_local: bool = False
    # ) -> AsyncIterator["AsyncMongoDBSaver"]:
    #     client = None
    #     try:
    #         if is_local:
    #             import certifi
    #             client = AsyncIOMotorClient(
    #                 host=host,
    #                 tlsCAFile=certifi.where()
    #             )
    #         else:
    #             if port is None:
    #                 client = AsyncIOMotorClient(host=host)
    #             else:
    #                 client = AsyncIOMotorClient(host=host, port=port)
    #         yield AsyncMongoDBSaver(
    #             client=client,
    #             database_name=database_name,
    #             collection_name=collection_name,
    #             ttl_index_name=ttl_index_name,
    #             ttl_index_key=ttl_index_key,
    #             ttl_expire_after_seconds=ttl_expire_after_seconds
    #         )
    #     finally:
    #         if client:
    #             client.close()

    async def create_index(
        self, *, name: str, expire_after_seconds: float
    ) -> None:
        while True:
            try:
                await self.collection.create_index(
                    keys=self.ttl_index_key,
                    name=name,
                    background=True,
                    expireAfterSeconds=expire_after_seconds
                )
            except OperationFailure:
                await self.collection.drop_index(name)
                continue
            break

    async def aget_tuple(
        self, config: RunnableConfig
    ) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the MongoDB database
        based on the provided config. If the config contains a "thread_ts"
        key, the checkpoint with the matching thread ID and timestamp is
        retrieved. Otherwise, the latest checkpoint for the given thread ID
        is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the
                checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or
                None if no matching checkpoint was found.
        """
        config_dict = config["configurable"]
        query = {
            k: config_dict.get(k, "") for k in ("thread_id", "checkpoint_ns")
        }
        if checkpoint_id := (
            config_dict.get("checkpoint_id") or config_dict.get("thread_ts")
        ) is not None:
            query["checkpoint_id"] = checkpoint_id
        result = (
            self.collection.find(query).sort("checkpoint_id", -1).limit(1)
        )
        async for doc in result:
            config_values = {
                k: doc[k]
                for k in ("thread_id", "checkpoint_ns", "checkpoint_id")
            }
            checkpoint = self.serde.loads_typed(
                (doc["type"], doc["checkpoint"])
            )
            serialized_writes = self.write_collection.find(config_values)
            pending_writes = [
                (
                    doc["task_id"],
                    doc["channel"],
                    self.serde.loads_typed((doc["type"], doc["value"])),
                )
                async for doc in serialized_writes
            ]
            if doc.get("parent_checkpoint_id"):
                parent_config = {
                    "configurable": {
                        **query,
                        "checkpoint_id": doc.get("parent_checkpoint_id")
                    }
                }
            else:
                parent_config = None
            return CheckpointTuple(
                config={"configurable": config_values},
                checkpoint=checkpoint,
                metadata=self.serde.loads(doc["metadata"]),
                parent_config=parent_config,
                pending_writes=pending_writes
            )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the MongoDB
        database based on the provided config. The checkpoints are ordered
        by checkpoint ID in descending order.

        Args:
            config (RunnableConfig): The config to use for listing the
                checkpoints.
            filter (dict[str, Any]): Additional filtering criteria for
                metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints
                before the specified checkpoint ID are returned. Defaults to
                None.
            limit (Optional[int]): The maximum number of checkpoints to
                return. Defaults to None.

        Yields:
            AsyncIterator[CheckpointTuple]: An Async iterator of checkpoint
            tuples.
        """
        query = {}
        if config is not None:
            conf_di = config["configurable"]
            query = {
                k: conf_di.get(k, "") for k in ("thread_id", "checkpoint_ns")
            }
        if filter:
            for key, value in filter.items():
                query[f"metadata.{key}"] = value
        if before is not None:
            query["checkpoint_id"] = {
                "$lt": before["configurable"]["checkpoint_id"]
            }
        result = self.collection.find(query).sort("checkpoint_id", -1)
        if limit is not None:
            result = result.limit(limit)
        async for doc in result:
            config = {
                "configurable": {
                    k: doc[k]
                    for k in ("thread_id", "checkpoint_ns", "checkpoint_id")
                }
            }
            checkpoint = self.serde.loads_typed(
                (doc["type"], doc["checkpoint"])
            )
            if doc.get("parent_checkpoint_id"):
                parent_config = {
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc["checkpoint_ns"],
                        "checkpoint_id": doc["parent_checkpoint_id"],
                    }
                }
            else:
                parent_config = None
            yield CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=self.serde.loads(doc["metadata"]),
                parent_config=parent_config
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata | None = None,
        new_versions: ChannelVersions | None = None,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the MongoDB database. The checkpoint
        is associated with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the
                checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with
                the checkpoint. Defaults to None.
            new_versions (ChannelVersions): New channel versions as of this
                write. Defaults to None.

        Returns:
            RunnableConfig: The updated config containing the saved
            checkpoint.
        """
        config_dict = config["configurable"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        doc = {
            "parent_checkpoint_id": config_dict.get("checkpoint_id"),
            "type": type_,
            "checkpoint": serialized_checkpoint,
            "metadata": self.serde.dumps(metadata),
            self.ttl_index_key: datetime.now(UTC),
        }
        filter = {
            "thread_id": config_dict["thread_id"],
            "checkpoint_ns": config_dict["checkpoint_ns"],
            "checkpoint_id": checkpoint["id"],
        }
        await self.collection.update_one(filter, {"$set": doc}, upsert=True)
        return {"configurable": filter}

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to
        the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each
                as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        config_dict = config["configurable"]
        thread_id = config_dict["thread_id"]
        checkpoint_ns = config_dict["checkpoint_ns"]
        checkpoint_id = config_dict["checkpoint_id"]
        operations = []
        for idx, (channel, value) in enumerate(writes):
            upsert_query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "idx": idx,
            }
            type_, serialized_value = self.serde.dumps_typed(value)
            operations.append(
                UpdateOne(
                    upsert_query,
                    {
                        "$set": {
                            "channel": channel,
                            "type": type_,
                            "value": serialized_value,
                            "created_at": datetime.now(UTC),
                        }
                    },
                    upsert=True,
                )
            )
        await self.write_collection.bulk_write(operations)
