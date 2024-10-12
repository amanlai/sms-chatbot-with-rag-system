from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_text_splitters.base import TextSplitter

REPLACER = {"’": "'", "\xa0": "", '“': '"', '”': '"'}


class BaseTextSplitterWithCleanup(TextSplitter):

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            content = doc.page_content
            for old, new in REPLACER.items():
                content = content.replace(old, new)
            texts.append(content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)


class CharacterTextSplitterWithCleanup(
    CharacterTextSplitter,
    BaseTextSplitterWithCleanup
):

    def __init__(
        self,
        *,
        separator: str | None = None,
        chunk_size: int | None = None,
        **kwargs: Any
    ):
        """Create a new TextSplitter."""
        self.separator = separator
        super().__init__(chunk_size=chunk_size, **kwargs)

    # def split_text(self, text: str) -> list[str]:
    #     """Split incoming text and return chunks."""
    #     # First we naively split the large input into a
    #     # bunch of smaller ones.
    #     splits = text.split(self.separator)
    #     return self._merge_splits(splits, self.separator)


class RecursiveCharacterTextSplitterWithCleanup(
    RecursiveCharacterTextSplitter,
    BaseTextSplitterWithCleanup
):

    def __init__(
        self,
        *,
        separators: list[str] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        **kwargs: Any
    ):
        """Create a new Recursive TextSplitter."""
        super().__init__(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
