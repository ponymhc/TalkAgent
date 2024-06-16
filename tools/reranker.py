from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.callbacks.manager import Callbacks
from sentence_transformers import CrossEncoder
from langchain.pydantic_v1 import Extra
from langchain.schema import Document
from typing import Optional, Sequence



class Reranker(BaseDocumentCompressor):
    model_name:str = None
    top_k: int = None
    model:CrossEncoder = None

    def __init__(
        self, 
        model_name: str = '/home/pony/workspace/nlp/pretrained_model/bge-reranker-base',
        top_k: int = 3
    ):
        super().__init__()
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def bge_rerank(self,query,docs):
        model_inputs =  [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[:self.top_k]


    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using BAAI/bge-reranker models.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.bge_rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results