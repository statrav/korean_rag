import importlib
import json

class RetrieverManager:
    def __init__(self, retriever_type, documents, embedding_model, chunk_size, chunk_overlap, top_k):
        self.retriever_type = retriever_type
        self.documents = documents
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.retriever = self._load_retriever()

    def _load_retriever(self):
        if self.retriever_type == "DenseRetriever":
            from .retrievers.dense_retriever import DenseRetriever
            return DenseRetriever(self.documents, self.embedding_model, self.chunk_size, self.chunk_overlap, self.top_k)
        elif self.retriever_type == "BM25Retriever":
            from .retrievers.bm25_retriever import BM25Retriever
            return BM25Retriever(self.documents)
        elif self.retriever_type == "EnsembleRetriever":
            from .retrievers.ensemble_retriever import EnsembleRetriever
            from .retrievers.dense_retriever import DenseRetriever
            from .retrievers.bm25_retriever import BM25Retriever
            dense = DenseRetriever(self.documents, self.embedding_model, self.chunk_size, self.chunk_overlap, self.top_k)
            bm25 = BM25Retriever(self.documents)
            return EnsembleRetriever([dense, bm25])
        else:
            raise ValueError(f"Unknown retriever type: {self.retriever_type}")

    def retrieve(self, query):
        return self.retriever.retrieve(query, self.top_k)