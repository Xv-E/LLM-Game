import os
from llama_cpp import Llama
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings('ignore')

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

    
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory

llm1 = LlamaCpp(model_path='ggml-model-Q4_K_M.gguf', temperature=0.0, n_threads = 30, n_gpu_layers = -1, n_ctx=2024)
#llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm1, 
    memory = memory,
    verbose=True
)


#x = llm1.predict(input="Hi, my name is Andrew")#conversation.predict(input="Hi, my name is Andrew")
#print(x)
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")
print(memory.buffer)

# memory.load_memory_variables({})

# memory = ConversationBufferMemory()

# memory.save_context({"input": "Hi"}, 
#                     {"output": "What's up"})

# print(memory.buffer)