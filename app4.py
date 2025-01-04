# from langchain.embeddings import Embeddings
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import numpy as np

# 1. 加载 SentenceTransformer 模型
model = SentenceTransformer('models/all-MiniLM-L6-v2')

# 2. 创建 LangChain 的 Embeddings 类
# class SentenceTransformersEmbedding(SentenceTransformer):
#     def __init__(self, model):
#         self.model = model

#     def embed(self, texts):
#         embeddings = self.model.encode(texts, show_progress_bar=True)
#         return embeddings

# # 3. 创建自定义嵌入对象
# embedding_model = SentenceTransformersEmbedding(model)

# 4. 创建 LLMChain 用于生成语言模型的响应
llm = LlamaCpp(model_path='ggml-model-Q4_K_M.gguf', temperature=0.5, n_threads = 30, n_gpu_layers = -1, n_ctx=2024, stop=["\nHuman:", "\nAI:"])

prompt_template = "this the chat history memory: {memory}, now you are a chatbot, answer question: {text}, make the answer breif, give the answer diertly, no multiple rounds of conversation"
prompt = PromptTemplate(input_variables=["memory", "text"], template=prompt_template)
chain = LLMChain(llm=llm, prompt=prompt)

# 5. 获取文本嵌入
texts = ["My father is Joe.", "my favorite fruit is apple.", "my favorite car is byd.", "my favorite car is byd."]
#embeddings = embedding_model.embed(texts)
embeddings = model.encode(texts)

query_text = "What is my favorite car?"
query_embedding = model.encode([query_text])

# Calculate cosine similarity between the query and all documents
cosine_scores = cosine_similarity(query_embedding, embeddings)

# Get the index of the most relevant document (highest cosine similarity)
most_similar_document_index = np.argmax(cosine_scores)
print("most_similar_document: ", texts[most_similar_document_index])
# 7. 使用 LLMChain 来生成响应
inputs = {
    "memory": texts[most_similar_document_index],
    "text": query_text
}
response = chain.run(inputs)
print("LLM Response:", response)

