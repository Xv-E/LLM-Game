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
               temperature=0.3, 
               n_threads = 30, 
               n_gpu_layers = -1, 
               n_ctx=2024, 
               stop=["\nHuman:", "\nAI:"])


from langchain.evaluation.qa import QAGenerateChain
from langchain.evaluation.qa import QAEvalChain
example_gen_chain = QAGenerateChain.from_llm(llm)

qa_eval_chain = QAEvalChain.from_llm(llm, prompt = prompt)
 
# 准备问答对及真实答案
examples = [
    {
        "query": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
        "ground_truth": "Paris"
    },
    {
        "query": "What is the square root of 16?",
        "answer": "The square root of 16 is 4.",
        "ground_truth": "4"
    }
]
predictions = [
    {
        "result": "Paris",
    },
    {
        "result": "7"
    }
]

# 对问答对进行评估
results = qa_eval_chain.evaluate(examples, predictions, question_key="query", answer_key="answer", prediction_key="result")

# 打印评估结果
for i, result in enumerate(results):
    print(f"Example {i + 1}:")
    print(f"  Question: {examples[i]['query']}")
    print(f"  Answer: {examples[i]['answer']}")
    print(f"  Ground Truth: {examples[i]['ground_truth']}")
    print(f"  Feedback: {result['results']}")