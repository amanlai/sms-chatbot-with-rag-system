from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.utils import make_serializable

# from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.collection import Collection
from ...utils.config import (
    RUN_EXACT_NEAREST_NEIGHBOR_VECTOR_SEARCH,
    USE_LLAMA_INDEX
)


if USE_LLAMA_INDEX:
    from langchain_core.vectorstores import VectorStore
    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch as LlamaMongoDBAtlasVectorSearch  # noqa: E501



def vector_search_stage(
    query_vector: list[float],
    search_field: str,
    index_name: str,
    top_k: int = 4,
    filter: dict[str, Any] | None = None,
    oversampling_factor: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Vector Search Stage without Scores.

    Scoring is applied later depending on strategy.
    vector search includes a vectorSearchScore that is typically used.
    hybrid uses Reciprocal Rank Fusion.

    Args:
        query_vector: List of embedding vector
        search_field: Field in Collection containing embedding vectors
        index_name: Name of Atlas Vector Search Index tied to Collection
        top_k: Number of documents to return
        oversampling_factor: this times limit is the number of candidates
        filter: MQL match expression comparing an indexed field.
            Some operators are not supported.
            See `vectorSearch filter docs <https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/#atlas-vector-search-pre-filter>`_


    Returns:
        Dictionary defining the $vectorSearch
    """  # noqa: E501
    if RUN_EXACT_NEAREST_NEIGHBOR_VECTOR_SEARCH:
        stage = {
            "exact": RUN_EXACT_NEAREST_NEIGHBOR_VECTOR_SEARCH,
            "index": index_name,
            "path": search_field,
            "queryVector": query_vector,
            "limit": top_k,
        }
    else:
        stage = {
            "index": index_name,
            "path": search_field,
            "queryVector": query_vector,
            "numCandidates": top_k * oversampling_factor,
            "limit": top_k,
        }
    if filter:
        stage["filter"] = filter
    return {"$vectorSearch": stage}


class MongoDBAtlasVectorSearchWithENN(MongoDBAtlasVectorSearch):

    def __init__(
        self,
        collection: Collection[dict[str, Any]],
        embedding: Embeddings,
        index_name: str = "vector_index",
        text_key: str = "text",
        embedding_key: str = "embedding",
        relevance_score_fn: str = "cosine",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            collection=collection,
            embedding=embedding,
            index_name=index_name,
            text_key=text_key,
            embedding_key=embedding_key,
            relevance_score_fn=relevance_score_fn,
            **kwargs,
        )

    def _similarity_search_with_score(
        self,
        query_vector: list[float],
        k: int = 4,
        pre_filter: dict[str, Any] | None = None,
        post_filter_pipeline: list[dict] | None = None,
        oversampling_factor: int = 10,
        include_embeddings: bool = False,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Core search routine. See external methods for details."""

        # Atlas Vector Search, potentially with filter
        pipeline = [
            vector_search_stage(
                query_vector,
                self._embedding_key,
                self._index_name,
                k,
                pre_filter,
                oversampling_factor,
                **kwargs,
            ),
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
        ]

        # Remove embeddings unless requested.
        if not include_embeddings:
            pipeline.append({"$project": {self._embedding_key: 0}})
        # Post-processing
        if post_filter_pipeline is not None:
            pipeline.extend(post_filter_pipeline)

        # Execution
        cursor = self._collection.aggregate(pipeline)  # type: ignore[arg-type]
        docs = []

        # Format
        for res in cursor:
            text = res.pop(self._text_key)
            score = res.pop("score")
            make_serializable(res)
            docs.append((Document(page_content=text, metadata=res), score))
        return docs


if USE_LLAMA_INDEX:

    class MongoDBAtlasVectorSearch(LlamaMongoDBAtlasVectorSearch, VectorStore):

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
