from os import getenv, remove
from tempfile import NamedTemporaryFile
from typing import Any

from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pymongo.collection import Collection

from streamlit.runtime.uploaded_file_manager import UploadedFile

from .config import DEBUG, LOCAL
from .ingest import IngestData

load_dotenv()

MONGO_URI = getenv("MONGO_URI")


def init_connection() -> MongoClient | None:
    try:
        if LOCAL:
            import certifi
            mongo_client = MongoClient(
                MONGO_URI,
                tlsCAFile=certifi.where()
            )
        else:
            mongo_client = MongoClient(MONGO_URI)
        if DEBUG:
            print("Connection to MongoDB successful")
    except ConnectionFailure as e:
        import streamlit as st
        st.write("Connection to the database was not successful. "
                 "Please contact IT support.")
        if DEBUG:
            print(f"Connection failed: {e}")
        return
    return mongo_client


def ingest_(*, collection: Collection, key: str, value: Any) -> None:
    """
    Ingest data into MongoDB
    """
    collection.insert_one({key: value})


def update_(*, collection: Collection, key: str, value: Any) -> None:
    """
    Update certain key-value pairs in documents in a collection
    """
    collection.update_one(
        {key: {"$exists": True}},
        {"$set": {key: value}}
    )


def index_document(
    *,
    collection: Collection,
    exists: bool,
    uploaded_file: UploadedFile
) -> None:
    # writing the file from RAM to a temporary file that is deleted later
    with NamedTemporaryFile(delete=False) as tmp:
        data_processor = IngestData(collection=collection)
        tmp.write(uploaded_file.read())
        data_processor.build_embeddings(
            filename=tmp.name,
            exists=exists
        )
    remove(tmp.name)


def add_to_collection(
    *,
    collection: Collection,
    key: str | bool,
    value: UploadedFile
) -> None:
    if isinstance(key, bool):
        index_document(
            collection=collection,
            exists=key,
            uploaded_file=value
        )
    else:
        if next(collection.find({key: {"$exists": True}}), None) is not None:
            update_(
                collection=collection,
                key=key,
                value=value
            )
        else:
            ingest_(
                collection=collection,
                key=key,
                value=value
            )
