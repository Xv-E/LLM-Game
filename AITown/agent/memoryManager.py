from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import faiss
import sys
import os
#import numpy as np
# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


class MemoryManager:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        embedding_dim = self.embedding_model.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.docstore = InMemoryDocstore({})
        self.index_to_docstore_id = {}
        self.vectorstore = FAISS(
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id=self.index_to_docstore_id,
            embedding_function=embedding_model  # ✅ 现在传的是 Embeddings 实例
        )
        self.clues = {}

    def add_task_clue(self, task: str, clue: str):
        """添加任务线索"""
        if task not in self.clues:
            self.clues[task] = []
        self.clues[task].append(clue)

    def get_task_clue(self, task: str) -> List[str]:
        """获取任务线索"""
        return self.clues.get(task, [])

    def add_memory(self, content: str, keys: list[str] = [], doc_id: str = None):
        doc = Document(
            page_content=content,
            metadata={
                "keys": keys
            }
        )
        self.vectorstore.add_documents([doc], ids=[doc_id] if doc_id else None)


    def update_memory(self, doc_id: str, keys: list[str], new_content: str):
        """通过 key 更新记忆内容（先删后加）"""
        # 删除旧内容（如果存在）
        if doc_id in self.docstore._dict:
            self.vectorstore.delete([doc_id])
        # 添加新内容
        new_doc = Document(page_content=new_content, metadata={"keys": keys})
        self.vectorstore.add_documents([new_doc], ids=[doc_id])

    def delete_memory_by_key(self, doc_id: str):
        self.vectorstore.delete([doc_id])

    def get_memories_by_key(self, key: str) -> str:
        returns = []
        for doc in self.docstore._dict.values():
            if  key in doc.metadata.get("keys"):
                returns.append(doc.page_content)
        return returns


    def query_memory(self, query: str = None, k = 10):
        docs = self.vectorstore.similarity_search(query, k=k)  # k 为返回文档的数量
        return [doc.page_content for doc in docs] if docs else []

import utility.embeddings as embeddings
if __name__ == "__main__":
    # 初始化
    embedding_model = embeddings.SentenceTransformerEmbeddings(path='models/all-MiniLM-L6-v2')
    manager = MemoryManager(embedding_model)


    # 更新记忆
    manager.add_memory(
        "Alicia 是一名旅店老板，喜欢听冒险故事，最近还学会了魔法", ["alicia", "旅店老板"], "alicia_description", 
    )
    manager.add_memory(
        "Alicia 住在一个小镇的旅店里，旅店的名字叫做“冒险者之家”", ["alicia"], "alicia_location", 
    )
    # 查询
    print(manager.get_memories_by_key("旅店老板"))

    # 删除
    #manager.delete_memory_by_key("npc:alicia")

    print(manager.query_memory("火球术"))