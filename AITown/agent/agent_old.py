from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import AITown.utility.prompt_template as pt
import AITown.utility.model_instances as mi

default_model_path = 'ggml-model-Q4_K_M.gguf'
default_embedding_model_path = 'models/all-MiniLM-L6-v2'

class CharacterAgent:
    def __init__(self, id, role_settings, model_path=None, embedding_model_path=None):
        self.id = id  # 唯一标识符
        self.role_settings = role_settings
        self.llm = mi.get_llama_instance()  # 使用单例获取 Llama 实例
        self.embeddings = mi.get_embedding_instance()  # 单例 Embedding

        # 初始化记忆存储
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

        # 初始化动作集合
        self.actions = {
            "move": self.move,
            "pickup": self.pickup,
            "attack": self.attack,
            "use_item": self.use_item
        }

    def retrieve_relevant_memory(self, query, other_agent_id=None, k=5):
        """检索记忆，可以指定与某个代理的交互记录"""
        if other_agent_id:
            query = f"from {other_agent_id}: {query}"
        query_vector = self.embeddings.embed_query(query)
        return self.vectorstore.similarity_search_by_vector(query_vector, k=k)

    def store_memory(self, query, response, other_agent_id=None):
        """存储记忆，可以指定对话对象"""
        self.vectorstore.add_texts([f"{other_agent_id}: {query}", f"{self.id}: {response}"])

    def perceive_environment(self, environment):
        """感知环境并生成描述"""
        # 假设 environment 是一个字典，包含地图状态和周围信息
        perception = f"当前环境：{environment}"
        self.store_memory("perceive_environment", perception)
        return perception

    def decide_action(self, query, environment):
        """基于环境和查询生成行动建议"""
        relevant_memory = self.retrieve_relevant_memory(query, k=3)
        memory_context = [doc.page_content for doc in relevant_memory]

        cur_prompt = pt.character_PT.format(
            role_settings=self.role_settings,
            memory=memory_context,
            environment=environment,
            query=query
        )
        print(f"[{self.id}] cur_prompt: {cur_prompt}")
        response = self.llm.invoke(cur_prompt)
        self.store_memory(query, response)
        return response

    def execute_action(self, action, params):
        """执行指定的动作"""
        if action in self.actions:
            return self.actions[action](**params)
        else:
            return f"未知动作: {action}"

    # 动作实现
    def move(self, direction, steps=1):
        return f"{self.id} 移动 {steps} 步，方向：{direction}"

    def pickup(self, item):
        return f"{self.id} 拾取了物品：{item}"

    def attack(self, target):
        return f"{self.id} 攻击了目标：{target}"

    def use_item(self, item, target=None):
        return f"{self.id} 使用了物品：{item}" + (f"，目标：{target}" if target else "")

    def generate_response(self, query, environment=None):
        """生成响应并根据建议路由动作"""
        action_suggestion = self.decide_action(query, environment)
        try:
            action_data = eval(action_suggestion)  # 假设建议返回 JSON 格式
            action = action_data.get("action")
            params = action_data.get("params", {})
            return self.execute_action(action, params)
        except Exception as e:
            return f"解析动作失败: {str(e)}"

    def interact_with(self, other_agent, query):
        """与另一个代理交互"""
        print(f"[{self.id}] to [{other_agent.id}]: {query}")
        response = other_agent.generate_response(query, environment=None)
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


    environment = {
    "position": (3, 3),
    "visible_tiles": [
        {"position": (3, 4), "type": "item", "content": "gold_coin"},
        {"position": (4, 3), "type": "enemy", "content": {"name": "goblin", "hp": 10}}
        ]
    }
    query = "当前情况下一步如何行动？"
    response = alice.generate_response(query, environment)
    print(response)
