from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, path: str):
        self.model = SentenceTransformer(path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """生成多个文档的嵌入向量"""
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """生成单个查询的嵌入向量"""
        embedding = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()