from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        """生成文档的嵌入向量"""
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        """生成单个查询的嵌入向量"""
        return self.model.encode([text], show_progress_bar=False).tolist()[0]

# 1. 加载 SentenceTransformer 模型
embedding_model = SentenceTransformer('models/all-MiniLM-L6-v2')
embedding_model = SentenceTransformerEmbeddings(embedding_model)

# 2. 准备文档数据
documents = [
    Document(page_content="My favorite fruit is apple."),
    Document(page_content="My favorite car is bmw."),
    Document(page_content="My favorite color is blue."),
    Document(page_content="My favorite singer is jay."),
    Document(page_content="I like Android"),
]

# 3. 生成嵌入向量
texts = [doc.page_content for doc in documents]
embeddings = embedding_model.model.encode(texts)

vectorstore = FAISS.from_documents(documents, embedding_model)

# vectorstore = FAISS(embedding_function=lambda x: embedding_model.encode(x).tolist())
# vectorstore.add_texts(texts, embeddings=embeddings)

prompt_template = """
You are a helpful assistant. Given the following context, answer the question below.
Context: {context}
Question: {question}
Answer:
"""

# 5. Create the prompt template using the custom template
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# 6. Create the chain with the custom prompt template
#qa_chain = LLMChain(llm=llm, prompt=prompt)

# 7. Set up the retriever and QA chain
retriever = vectorstore.as_retriever()

llm = LlamaCpp(model_path='ggml-model-Q4_K_M.gguf', 
               temperature=0.5, 
               n_threads = 30, 
               n_gpu_layers = -1, 
               n_ctx=2024, 
               stop=["\nHuman:", "\nAI:"], 
               prompt=prompt)

qa_chain = RetrievalQA.from_chain_type(
    llm =llm,
    chain_type="stuff", 
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)



# 定义查询问题
question = "What is my favorite phone? iphone or huawei?"

# 使用 RetrievalQA 生成回答
result = qa_chain.run(question)

# 输出结果
print("Answer:", result)
