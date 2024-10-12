#! ../venv/Scripts/python.exe
import logging
from time import sleep

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader

from langchain_core.documents import Document

from langchain_mongodb import MongoDBAtlasVectorSearch

from langchain_openai import OpenAIEmbeddings

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
    USE_RECURSIVE_SPLITTER,
)

from .text_splitter import (
    CharacterTextSplitterWithCleanup,
    RecursiveCharacterTextSplitterWithCleanup
)

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
                    "path": "page",
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
        self.embeddings = OpenAIEmbeddings(disallowed_special=(), model=MODEL_NAME)  # noqa: E501
        self.collection = collection

    def load_document(self, *, filename: str) -> list[Document]:
        """
        Load PDF files as LangChain Documents
        """
        documents = None
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
        if DEBUG:
            print("\n\n\nChunking complete...\n")
            print(f"{len(chunks)} chunks were created.\n")
        # if data already exists in the collection
        # remove all documents in that collection
        if exists:
            self.collection.drop()
        # create document collection
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
        # if not exists:
        #     self.collection.create_search_index(model=search_index_model)
        # else:
        #     # if an index with the same name is created, it may cause
        #     # Duplicate Index error if the previous collection takes some
        #     # time to be removed while loop would help make sure to create
        #     # the search index
        #     err = 1
        #     while err is not None:
        #         try:
        #             self.collection.create_search_index(
        #                 model=search_index_model
        #             )
        #             err = None
        #         except OperationFailure as e:
        #             sleep(2)
        #             if not str(e).startswith("Duplicate Index"):
        #                 import streamlit as st
        #                 st.write("An unexpected error occurred. "
        #                          "Please contact IT support.")
        #                 err = None
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
