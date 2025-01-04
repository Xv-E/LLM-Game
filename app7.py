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
You are a helpful assistant. evalue the Model result briefly and directly, don't continue generating other questions and answers!
no any other additional response!
Question: {query}
Reference Answer: {answer}
Model result: {result}

give Feedback that the result is right or wrong.
and give a final score from 0 to 10, the higher score the more accurate.
"""
# 5. Create the prompt template using the custom template
prompt = PromptTemplate(input_variables=["query", "answer", "result"], template=prompt_template)

# 6. Create the chain with the custom prompt template
#qa_chain = LLMChain(llm=llm, prompt=prompt)

# 7. Set up the retriever and QA chain
retriever = vectorstore.as_retriever()

llm = LlamaCpp(model_path='ggml-model-Q4_K_M.gguf', 
               temperature=0.4, 
               n_threads = 30, 
               n_gpu_layers = -1, 
               n_ctx=2024,)

from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.agents import AgentType

# Initialize the PythonREPLTool
# python_tool = PythonREPLTool()

# agent = create_python_agent(
#     llm,
#     tool=python_tool,
#     verbose=True
# )

# # 4. 使用代理
# question = "What is 25 * 4 + 10"
# response = agent.invoke(question)
# print(response)

def calculate_flight_duration(origin: str, destination: str):
    try:
        return  f"The flight duration from {origin} to {destination} is approximately 22 hours."
    except ValueError:
        return "Invalid input format. Please provide input as 'City1, City2'."
    
flight_tool = Tool(
    name="calculate_flight_duration",
    func=lambda input: calculate_flight_duration(*map(str.strip, input.split(","))),
    description=(
        "Calculates flight duration between two cities. "
        "Input should be two city names separated by a comma, e.g., 'New York, Los Angeles'."
    ),
)

tools = [flight_tool]
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True,
                         handle_parsing_errors = True,
                         max_iterations=3,
                         output_key = "output",
                         )

# 调用代理
response = agent.invoke({"input": "Calculate flight duration from Tokyo to Los Angeles"}, stop=["Final Answer", "\n"])
print("response:", response["output"])
# question = "How long is the flight from Tokyo to Sydney?"
# response = executor.run(question)
# print(response)
