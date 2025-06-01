from typing import List, Dict
import sys
import os
#import numpy as np
# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from utility import prompt_template as pt
from utility import model_instances as mi
from agent.memoryManager import MemoryManager

# CharacterAgent: 角色
class CharacterAgent:
    def __init__(self, name: str, personality: str, memory_manager: MemoryManager, prompt_template: str, llm):
        self.name = name
        self.personality = personality
        self.memory_manager = memory_manager
        self.prompt_template = prompt_template
        self.llm = llm

    def generate_response(self, context: str, query: str):
        """生成对话回复"""
        relevant_memories = self.memory_manager.retrieve_memory(query, k=2)
        memory_text = "\n".join(relevant_memories)

        # 根据模板生成完整 prompt
        prompt = self.prompt_template.format(
            personality=self.personality,
            memory=memory_text,
            context=context,
            query=query
        )
        response = self.llm.invoke(prompt)

        # 存储到记忆
        self.memory_manager.store_memory(f"Human: {query}")
        self.memory_manager.store_memory(f"{self.name}: {response}")
        return response

# ChatRoom: 聊天室
class ChatRoom:
    def __init__(self):
        self.roles: List[CharacterAgent] = []
        self.history: List[str] = []
        self.current_index = 0

    def add_role(self, role: CharacterAgent):
        """添加角色"""
        self.roles.append(role)

    def next_turn(self, query: str = None):
        """角色轮流发言"""
        if not self.roles:
            print("聊天室中没有角色！")
            return
        
        current_role = self.roles[self.current_index]
        context = "\n".join(self.history)

        # 角色生成响应
        if query:
            self.history.append(f"Human: {query}")
        response = current_role.generate_response(context, query or "")
        self.history.append(f"{current_role.name}: {response}")
        print(f"{current_role.name}: {response}")

        # 切换到下一个角色
        self.current_index = (self.current_index + 1) % len(self.roles)

    def print_history(self):
        """打印历史对话"""
        print("\n对话记录：")
        for entry in self.history:
            print(entry)

# 示例
import utility.embeddings as embeddings
from langchain_community.llms import LlamaCpp

if __name__ == "__main__":
    # 初始化
    embedding_model = embeddings.SentenceTransformerEmbeddings(path='models/all-MiniLM-L6-v2')
    llm = mi.get_llama_instance()
    # llm = LlamaCpp(
    #     model_path="ggml-model-q4_0.bin",
    #     temperature=0.7,
    #     n_threads=8,
    #     n_ctx=1024,
    # )
    
    # 创建角色
    warrior_memory = MemoryManager(embedding_model)
    mage_memory = MemoryManager(embedding_model)

    warrior = CharacterAgent(
        name="Warrior",
        personality="Brave and straightforward, always eager to fight.",
        memory_manager=warrior_memory,
        prompt_template="You are a {personality} warrior. Here is the context:\n{context}\nYour memory:\n{memory}\nRespond to: {query}",
        llm=llm,
    )

    mage = CharacterAgent(
        name="Mage",
        personality="Wise and calm, prefers magic over brute force.",
        memory_manager=mage_memory,
        prompt_template="You are a {personality} mage. Here is the context:\n{context}\nYour memory:\n{memory}\nRespond to: {query}",
        llm=llm,
    )

    # 创建聊天室
    chatroom = ChatRoom()
    chatroom.add_role(warrior)
    chatroom.add_role(mage)

    # 模拟对话
    chatroom.next_turn("What do you think about the enemy?")
    chatroom.next_turn()
    chatroom.next_turn("What should we do next?")
    chatroom.print_history()