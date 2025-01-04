from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.llms import LlamaCpp
import faiss
import prompt_template as pt
import model_instances as mi

class CharacterAgent:
    def __init__(self, model_path, embedding_model_path, role_settings):
        self.role_settings = role_settings
        self.role_name = role_settings.get("name", "Unknown")
        self.llm = mi.get_llama_instance(model_path)

        # 初始化 SentenceTransformer 嵌入
        self.embeddings = mi.get_embedding_instance(embedding_model_path)
        embedding_dim = self.embeddings.model.get_sentence_embedding_dimension()

        # 初始化 FAISS 索引
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.docstore = InMemoryDocstore({})
        self.index_to_docstore_id = {}

        self.vectorstore = FAISS(
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id=self.index_to_docstore_id,
            embedding_function=self.embeddings.embed_query
        )

    def retrieve_relevant_memory(self, query, target_role=None, k=5):
        """
        检索与 query 相关的记忆，支持按角色过滤。
        """
        query_vector = self.embeddings.embed_query(query)
        results = self.vectorstore.similarity_search_by_vector(query_vector, k=k)

        if target_role:
            # 根据角色名过滤记忆
            results = [doc for doc in results if target_role in doc.page_content]

        return results

    def store_memory(self, human, ai, target_role=None):
        """
        存储交互内容，包含角色标识。
        """
        role_prefix = f"[{self.role_name}]" if not target_role else f"[{target_role}]"
        human_entry = f"{role_prefix} Human: {human}"
        ai_entry = f"{role_prefix} AI: {ai}"
        self.vectorstore.add_texts([human_entry, ai_entry])

    def generate_response(self, query, target_role=None):
        relevant_memory = self.retrieve_relevant_memory(query, target_role=target_role, k=2)
        relevant_memory_str = [doc.page_content for doc in relevant_memory]

        cur_prompt = pt.character_PT.format(
            role_settings=self.role_settings,
            memory=relevant_memory_str,
            query=query
        )
        print("cur_prompt", cur_prompt)
        response = self.llm.invoke(cur_prompt)
        self.store_memory(query, response, target_role=self.role_name)
        return response


if __name__ == "__main__":
    # 创建两个角色
    role_alice = {
        "name": "Alice",
        "personality": "Friendly and curious.",
        "hobbies": "reading, traveling, coding",
        "backstory": "An adventurer seeking knowledge."
    }
    role_bob = {
        "name": "Bob",
        "personality": "Calm and analytical.",
        "hobbies": "chess, philosophy, programming",
        "backstory": "A philosopher who enjoys deep thinking."
    }

    agent_alice = CharacterAgent(
        model_path="ggml-model-Q4_K_M.gguf",
        embedding_model_path='models/all-MiniLM-L6-v2',
        role_settings=role_alice
    )

    agent_bob = CharacterAgent(
        model_path="ggml-model-Q4_K_M.gguf",
        embedding_model_path='models/all-MiniLM-L6-v2',
        role_settings=role_bob
    )

    # Alice 与用户交互
    response_alice = agent_alice.generate_response("Hi Alice, what do you enjoy?")
    print(f"Alice: {response_alice}")

    # 用户转向与 Bob 交互
    response_bob = agent_bob.generate_response("Hi Bob, what's your favorite hobby?")
    print(f"Bob: {response_bob}")

    # Alice 提及 Bob
    response_alice_about_bob = agent_alice.generate_response("Alice, do you know Bob?")
    print(f"Alice about Bob: {response_alice_about_bob}")
