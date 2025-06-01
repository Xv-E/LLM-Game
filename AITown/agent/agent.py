from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import AITown.utility.prompt_template as pt
import AITown.utility.model_instances as mi

default_model_path='ggml-model-Q4_K_M.gguf'
default_embedding_model_path='models/all-MiniLM-L6-v2'

class CharacterAgent:
    def __init__(self, id, role_settings, model_path=None, embedding_model_path=None):
        self.id = id  # 唯一标识符
        self.role_settings = role_settings
        self.llm = mi.get_llama_instance()  # 使用单例获取 Llama 实例
        self.embeddings = mi.get_embedding_instance()  # 单例 Embedding

        # initial vectorstore
        embedding_dim = self.embeddings.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.docstore = InMemoryDocstore({})
        self.index_to_docstore_id = {}
        self.vectorstore = FAISS(
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id=self.index_to_docstore_id,
            embedding_function=self.embeddings.embed_query
        )

    def retrieve_relevant_memory(self, query, other_agent_id=None, k=5):
        """检索记忆，可以指定与某个代理的交互记录"""
        if other_agent_id:
            query = f"from {other_agent_id}: {query}"
        query_vector = self.embeddings.embed_query(query)
        return self.vectorstore.similarity_search_by_vector(query_vector, k=k)

    def store_memory(self, query, response, other_agent_id=None):
        """存储记忆，可以指定对话对象"""
        self.vectorstore.add_texts([f"{other_agent_id}: {query}", f"{self.id}: {response}"])

    def generate_response(self, query, other_agent_id=None):
        """生成响应，支持与特定代理的对话上下文"""
        relevant_memory = self.retrieve_relevant_memory(query, other_agent_id, k=2)
        relevant_memory_str = [doc.page_content for doc in relevant_memory]

        cur_prompt = pt.character_PT.format(
            role_settings=self.role_settings,
            memory=relevant_memory_str,
            query=query
        )
        print(f"[{self.id}] cur_prompt: {cur_prompt}")
        response = self.llm.invoke(cur_prompt)
        self.store_memory(query, response, other_agent_id)
        return response

    def interact_with(self, other_agent, query):
        """与另一个代理交互"""
        print(f"[{self.id}] to [{other_agent.id}]: {query}")
        response = other_agent.generate_response(query, other_agent_id=self.id)
        print(f"[{other_agent.id}] to [{self.id}]: {response}")
        return response
    

if __name__ == "__main__":
    # 角色设定
    alice_role = {
        "name": "Alice",
        "personality": "Friendly and curious, loves to ask questions.",
        "hobbies": "Reading, cooking, painting",
        "backstory": "An artist who loves exploring different cultures."
    }
    bob_role = {
        "name": "Bob",
        "personality": "Analytical and logical, enjoys solving problems.",
        "hobbies": "Chess, programming, hiking",
        "backstory": "A software engineer with a passion for puzzles."
    }

    # 初始化代理
    alice = CharacterAgent(
        id="Alice",
        role_settings=alice_role
    )
    bob = CharacterAgent(
        id="Bob",
        role_settings=bob_role
    )

    # 代理间对话
    query = "Hi Bob, what do you think about chess?"
    response = alice.interact_with(bob, query)

    # Bob 继续与 Alice 互动
    query = "Hi Alice, do you like chess too?"
    response = bob.interact_with(alice, query)

