#!/usr/bin/env python
# coding: utf-8

# # LangChain: Q&A over Documents
# 
# An example might be a tool that would allow you to query a product catalog for items of interest.

# In[ ]:


#pip install --upgrade langchain


# In[ ]:


import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


# Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video.

# In[ ]:


# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"


# In[ ]:


from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
# from langchain.llms import OpenAI
from langchain_community.llms import LlamaCpp

# In[ ]:


#file = 'OutdoorClothingCatalog_1000.csv'
#loader = CSVLoader(file_path=file)


# In[ ]:


from langchain.indexes import VectorstoreIndexCreator
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
# In[ ]:


#pip install docarray


# In[ ]:

documents = [
    Document(page_content="This is a document about AI."),
    Document(page_content="This document discusses machine learning."),
    Document(page_content="Here we talk about natural language processing."),
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_creator = VectorstoreIndexCreator(vectorstore_cls=FAISS, embedding_function=embeddings)
index = index_creator.from_documents(documents)
# index = VectorstoreIndexCreator(
#     vectorstore_cls=DocArrayInMemorySearch
# ).from_loaders([loader])


# In[ ]:


query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."


# **Note**:
# - The notebook uses `langchain==0.0.179` and `openai==0.27.7`
# - For these library versions, `VectorstoreIndexCreator` uses `text-davinci-003` as the base model, which has been deprecated since 1 January 2024.
# - The replacement model, `gpt-3.5-turbo-instruct` will be used instead for the `query`.
# - The `response` format might be different than the video because of this replacement model.

# In[ ]:

llm_replacement_model = LlamaCpp(model_path='ggml-model-Q4_K_M.gguf', temperature=0.5, n_threads = 30, n_gpu_layers = -1, n_ctx=2024, stop=["\nHuman:", "\nAI:"])
llm = LlamaCpp(model_path='ggml-model-Q4_K_M.gguf', temperature=0.5, n_threads = 30, n_gpu_layers = -1, n_ctx=2024, stop=["\nHuman:", "\nAI:"])
# llm_replacement_model = OpenAI(temperature=0, model='gpt-3.5-turbo-instruct')

response = index.query(query, llm = llm_replacement_model)


# In[ ]:


display(Markdown(response))


# ## Step By Step

# In[ ]:


from langchain.document_loaders import CSVLoader
#loader = CSVLoader(file_path=file)


# In[ ]:


#docs = loader.load()


# In[ ]:


#docs[0]


# In[ ]:


from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()


# In[ ]:


embed = embeddings.embed_query("Hi my name is Harrison")


# In[ ]:


print(len(embed))


# In[ ]:


print(embed[:5])


# In[ ]:


db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)


# In[ ]:


query = "Please suggest a shirt with sunblocking"


# In[ ]:


docs = db.similarity_search(query)


# In[ ]:


len(docs)


# In[ ]:


docs[0]


# In[ ]:


retriever = db.as_retriever()


# In[ ]:


# llm = ChatOpenAI(temperature = 0.0, model=llm_model)


# In[ ]:


qdocs = "".join([docs[i].page_content for i in range(len(docs))])


# In[ ]:


response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 


# In[ ]:


display(Markdown(response))


# In[ ]:


qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)


# In[ ]:


query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."


# In[ ]:


response = qa_stuff.run(query)


# In[ ]:


display(Markdown(response))


# In[ ]:


response = index.query(query, llm=llm)


# In[ ]:


index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])


# Reminder: Download your notebook to you local computer to save your work.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




