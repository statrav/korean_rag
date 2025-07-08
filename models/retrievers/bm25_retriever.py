from typing import List

class BM25Retriever:
    def __init__(self, documents: List[str]):
        # 실제 BM25 인덱스 구축 코드 필요
        self.documents = documents
    def retrieve(self, query: str, top_k: int = 3):
        # 실제 BM25 검색 로직 필요 (여기선 단순 예시)
        return self.documents[:top_k]
    def as_retriever(self):
        return self 