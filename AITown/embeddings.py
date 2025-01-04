from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, path):
        self.model = SentenceTransformer(path)

    def embed_documents(self, texts):
        """生成文档的嵌入向量"""
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        """生成单个查询的嵌入向量"""
        return self.model.encode([text], show_progress_bar=False).tolist()[0]
