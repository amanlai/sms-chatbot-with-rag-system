#! ../venv/Scripts/python.exe
import logging
from time import sleep

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader

from langchain_core.documents import Document

from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_SEPARATOR,
    INDEX_NAME,
    MODEL_NAME,
    DEBUG,
    USE_LLAMA_INDEX,
    USE_RECURSIVE_SPLITTER,
)

from .text_splitter import (
    CharacterTextSplitterWithCleanup,
    RecursiveCharacterTextSplitterWithCleanup
)

if USE_LLAMA_INDEX:
    from llama_index.core import (
        SimpleDirectoryReader,
        VectorStoreIndex,
        StorageContext
    )
    from llama_index.core.node_parser import LangchainNodeParser
    from llama_index.embeddings.openai import OpenAIEmbedding
    from .vector_stores import MongoDBAtlasVectorSearch
else:
    from langchain_mongodb import MongoDBAtlasVectorSearch
    from langchain_openai import OpenAIEmbeddings


logger = logging.getLogger(__name__)


class IngestData:

    VS_INDEX = {
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": None,
                    "similarity": "cosine",
                },
                {
                    "path": "metadata.page",
                    "type": "filter"
                },
            ],
        },
        "name": INDEX_NAME,
        "type": "vectorSearch",
    }

    def __init__(
            self,
            *,
            collection: Collection,
    ) -> None:
        # embedding function to be used in collection
        if USE_LLAMA_INDEX:
            self.embeddings = OpenAIEmbedding(model=MODEL_NAME)
        else:
            self.embeddings = OpenAIEmbeddings(disallowed_special=(), model=MODEL_NAME)  # noqa: E501
        self.collection = collection

    def load_document(self, *, filename: str) -> list[Document]:
        """
        Load PDF files as LangChain Documents
        """
        documents = None
        if USE_LLAMA_INDEX:
            loader = SimpleDirectoryReader(input_files=[filename])
            documents = loader.load_data()
        else:
            if filename.rsplit(".", 1)[-1] == "pdf":
                loader = PyPDFLoader(filename)
            else:
                loader = Docx2txtLoader(filename)
            try:
                documents = loader.load()
            except Exception as err:
                print(f"ERROR: {filename}", err)
        return documents

    def chunk_data(
        self, *, filename: str
    ) -> list[Document]:
        """
        Load the document, split it and return the chunks
        """
        # load document
        documents = self.load_document(filename=filename)
        if not documents:
            print("No new documents to load")
            exit(0)
        # split the document
        if USE_RECURSIVE_SPLITTER:
            text_splitter = RecursiveCharacterTextSplitterWithCleanup(
                separators=[CHUNK_SEPARATOR],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        else:
            text_splitter = CharacterTextSplitterWithCleanup(
                separator=CHUNK_SEPARATOR,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        if USE_LLAMA_INDEX:
            parser = LangchainNodeParser(text_splitter)
            chunks = parser.get_nodes_from_documents(documents)
        else:
            chunks = text_splitter.split_documents(documents)
        if DEBUG:
            print(f"Split into {len(chunks)} chunks of text"
                  f"(max. {CHUNK_SIZE} tokens each)")
        return chunks

    def build_embeddings(
        self, *, filename: str, exists: bool = False
    ) -> None:
        """
        Create embeddings and save them in a Chroma vector store
        Returns the indexed db
        """
        chunks = self.chunk_data(filename=filename)
        # if data already exists in the collection
        # remove all documents in that collection
        if exists:
            self.collection.drop()
        # create document collection
        if USE_LLAMA_INDEX:
            vector_store = MongoDBAtlasVectorSearch.from_documents(
                collection=self.collection,
            )
            vector_store_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            VectorStoreIndex(
                nodes=chunks,
                storage_context=vector_store_context,
                embed_model=self.embeddings
            )
        else:
            MongoDBAtlasVectorSearch.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection=self.collection,
                index_name=INDEX_NAME,
            )
        # create the search index model
        embedding_length_dict = self.collection.find_one(
            {},
            {"length": {"$size": "$embedding"}, "_id": False}
        )
        length = embedding_length_dict["length"]
        self.VS_INDEX["definition"]["fields"][0]["numDimensions"] = length
        search_index_model = SearchIndexModel(**self.VS_INDEX)
        # create vector search index
        while True:
            try:
                self.collection.create_search_index(
                    model=search_index_model
                )
            except OperationFailure as e:
                if not str(e).startswith("Duplicate Index"):
                    import streamlit as st
                    st.write("An unexpected error occurred. "
                             "Please contact IT support.")
                    exit(0)
                else:
                    sleep(2)
                    continue
            break
        if DEBUG:
            print("Ingestion complete!")
