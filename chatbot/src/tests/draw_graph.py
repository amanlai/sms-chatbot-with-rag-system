from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from agents.plan_executor import PlanExecutor


# example from
class ToyRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        matching_documents = []
        for document in self.documents:
            if len(matching_documents) > 3:
                return matching_documents

            if query.lower() in document.page_content.lower():
                matching_documents.append(document)
        return matching_documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError


def draw_graph():

    try:
        documents = [
            Document(
                page_content="Labor Day is a holiday that is celebrated "
                "on September 1 each year if it is a Monday."
            ),
            Document(
                page_content="September 1 this year fell on Sunday. So "
                "there was no Labor Day."
            ),
            Document(
                page_content="Next year we also don't have Labor Day. "
                "This is because September 1 will also not be on Monday."
            )
        ]
        retriever = ToyRetriever(documents=documents)

        graph = PlanExecutor(retriever=retriever).compile()
        img = graph.get_graph(xray=True)
        img.draw_mermaid_png(output_file_path="./static/rag_architecture.png")
    except Exception:
        raise
