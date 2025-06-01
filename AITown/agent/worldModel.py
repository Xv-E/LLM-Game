from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import sys
import os
#import numpy as np
# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from tool.generate_gbnf import add_terms, term_exist, EntityType

class WorldModel:
    npcs = {}
    locations = {}
    inventories = {}

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
            embedding_function=self.embedding_model.embed_query
        )

    def add_fact(self, fact: str):
        """加入世界知识，比如地点、NPC、事件、规则等"""
        self.vectorstore.add_texts([fact])

    def add_term(self, type:EntityType, term: str):
        """添加一个术语到世界知识中"""
        add_terms(type, term)

    def term_exist(self, type:EntityType, term: str):
        """检查一个术语是否存在于世界知识中"""
        return term_exist(type, term)

    def query_facts(self, query: str, k: int = 10):
        """根据自然语言问题检索相关世界知识"""
        query_vector = self.embedding_model.embed_query(query)
        results = self.vectorstore.similarity_search_by_vector(query_vector, k=k)
        return [doc.page_content for doc in results]
    
    def search(self, query: str) -> List[str]:
        """基于关键词检索返回描述（可嵌入 LLM prompt）"""
        results = []
        for npc in self.npcs:
            if any(keyword in query.lower() for keyword in npc.get_keywords()):
                results.append(npc.describe())
        return results

    def register_entity(self, npc):
        """注册一个 NPC，并记录其可检索的信息"""
        description = npc.describe()
        self.npcs[npc.name] = npc

        self.add_fact(description)  # 将 NPC 描述加入世界知识
        self.add_term(EntityType.PERSON, npc.name)
        self.locations[npc.location.name].add_npc(npc)

    def register_location(self, location):
        """注册一个 NPC，并记录其可检索的信息"""
        description = location.describe()

        self.add_fact(description)
        self.locations[location.name] = location
        self.add_term(EntityType.LOCATION, location.name)

    def register_inventory(self, inventory):
        description = inventory.describe()
        self.add_fact(description)
        self.inventories[inventory.name] = inventory
        self.add_term(EntityType.INVENTORY, inventory.name)

    def get_location_by_name(self, name: str):
        return self.locations.get(name, None)

    def get_npc_by_name(self, name: str):
        return self.npcs.get(name, None)
    
    def get_inventory_by_name(self, name: str):
        return self.inventories.get(name, None)

# 示例
import utility.embeddings as embeddings
if __name__ == "__main__":
    # 初始化
    embedding_model = embeddings.SentenceTransformerEmbeddings(path='models/all-MiniLM-L6-v2')
    world = WorldModel(embedding_model)

    # 添加世界知识
    world.add_fact("Android phones are popular in 2023.")
    world.add_fact("IOS system is brief and fast.")
    world.add_fact("Apple is a red or green fruit.")

    # 查询
    query = "I like Apple phone."
    results = world.query_facts(query, k=3)

    for fact in results:
        print(f"🌍 世界知识相关信息: {fact}")
