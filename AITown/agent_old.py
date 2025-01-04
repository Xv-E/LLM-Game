#from langchain.chains import ConversationChain
#from langchain.memory import ConversationSummaryBufferMemory
#from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.llms import LlamaCpp
import faiss
import numpy as np
import embeddings
import prompt_template as pt
import model_instances as mi

class CharacterAgent:
    def __init__(self, model_path, embedding_model_path, role_settings):
        self.role_settings = role_settings
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
        #self.vectorstore.add_texts(role_settings)
        
        # memory form langchain
        # self.memory = VectorStoreRetrieverMemory(
        #     retriever=self.vectorstore.as_retriever(),
        #     memory_variables=["input"],
        #     k=1,
        # )
        #self.memory.save_context({"input": "Who are you"}, {"role setting:": role_settings})
        #self.conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=True,)

    def retrieve_relevant_memory(self, query, k=5):
        query_vector = self.embeddings.embed_query(query)
        return self.vectorstore.similarity_search_by_vector(query_vector, k=k)

    def store_memory(self, human, ai):
        #self.memory.save_context({"human": human}, {"ai": ai})
        self.vectorstore.add_texts(['human:' + human, 'AI:' + ai])

    def generate_response(self, query):
        relevant_memory = self.retrieve_relevant_memory(query, k=2)
        relevant_memory_str = [doc.page_content for doc in relevant_memory]

        cur_prompt = pt.character_PT.format(
            role_settings=self.role_settings,
            memory=relevant_memory_str,
            query=query
        )
        print("cur_prompt", cur_prompt)
        response = self.llm.invoke(cur_prompt)
        self.store_memory(query, response)
        return response

# 使用示例
if __name__ == "__main__":
    # 角色设定
    role = {
        "name":         "Alice",
        "personality":  "Mean and negative, talk dirty, always start with 'hahahaa'",
        "hobbies":      "reading, traveling, coding",
        "backstory":    "A fat Liar."
    }

    # 初始化角色代理
    agent = CharacterAgent(model_path="ggml-model-Q4_K_M.gguf", embedding_model_path='models/all-MiniLM-L6-v2', role_settings=role)

    # 用户输入
    user_query = "hi, I am Mike. What is your name?"
    response = agent.generate_response(user_query)
    print(response)

    response = agent.generate_response("Tell me about your favorite hobby.")
    print(response)

    response = agent.generate_response("Hi, do you know who am I")
    print(response)

    # response = agent.generate_response("My favourite car is bmw.")
    # print(response)

    # response = agent.generate_response("My favourite bag is gucci.")
    # print(response)

    # response = agent.generate_response("My favourite shoe is nike.")
    # print(response)
    # response = agent.generate_response("My favourite shirt is HM.")
    # print(response)

    # response = agent.generate_response("My favourite singer is jay.")
    # print(response)

    # response = agent.generate_response("Hi, do you know my favourite car")
    # print(response)
    #summary = agent.memory.load_memory_variables({})
    #print('summary', summary)

