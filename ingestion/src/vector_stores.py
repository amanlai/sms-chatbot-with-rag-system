from typing import Any

from langchain_core.vectorstores import VectorStore

from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch as LlamaMongoDBAtlasVectorSearch  # noqa: E501


class MongoDBAtlasVectorSearch(LlamaMongoDBAtlasVectorSearch, VectorStore):

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
