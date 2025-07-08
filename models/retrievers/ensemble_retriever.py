class EnsembleRetriever:
    def __init__(self, retrievers):
        self.retrievers = retrievers
    def retrieve(self, query, top_k=3):
        results = []
        for retriever in self.retrievers:
            results.extend(retriever.retrieve(query, top_k))
        # 중복 제거 및 top_k 반환 (간단 예시)
        seen = set()
        unique_results = []
        for r in results:
            if str(r) not in seen:
                unique_results.append(r)
                seen.add(str(r))
            if len(unique_results) >= top_k:
                break
        return unique_results
    def as_retriever(self):
        return self 