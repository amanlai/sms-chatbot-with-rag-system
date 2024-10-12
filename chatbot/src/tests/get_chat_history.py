import asyncio
from typing import Any

from langchain_core.messages import BaseMessage

from utils import AnswerGenerator

conn = AnswerGenerator()


async def get_chat_history(
    *, collection_name: str, thread_id: str
) -> list[BaseMessage]:
    checkpointer = conn.get_checkpointer(collection_name=collection_name)
    config = {
        "configurable": {"thread_id": thread_id}
    }
    checkpoint = await checkpointer.aget_tuple(config)
    messages = checkpoint.checkpoint["channel_values"]["messages"]
    return messages


def get_documents_in_collection(
    *, database_name: str, collection_name: str, asynchronous=False
) -> list[dict[str, Any]]:
    collection = conn.checkpoint_client[database_name][collection_name]
    cursor = collection.find()
    if asynchronous:
        from motor.motor_asyncio import AsyncIOMotorCursor

        async def inner(cursor: AsyncIOMotorCursor):
            return await cursor.to_list(None)
        return asyncio.run(inner(cursor=cursor))
    else:
        return list(cursor)
