from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import json

llm = LlamaCpp(model_path='ggml-model-Q4_K_M.gguf', 
               temperature=0.4, 
               n_threads = 30, 
               n_gpu_layers = -1, 
               n_ctx=2024,
               grammar_path="grammar.gbnf",
               stop=[";"])

json_data = llm.invoke("mike is 18-year-old, and is a girl")
print(json_data)
python_obj = json.loads(json_data)
print(python_obj)
