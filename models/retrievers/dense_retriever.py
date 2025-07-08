from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

class DenseRetriever:
    def __init__(self, documents, embedding_model, chunk_size, chunk_overlap, top_k):
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.texts = self.text_splitter.create_documents(documents)
        self.vectorstore = FAISS.from_documents(self.texts, embedding_model)
        self.top_k = top_k

    def retrieve(self, query, top_k):
        return self.vectorstore.similarity_search(query, k=top_k)