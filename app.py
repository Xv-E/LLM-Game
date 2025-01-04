import streamlit as st
import numpy as np

from langchain_community.document_loaders import CSVLoader
#from langchain.vectorstores import FAISS   llx-BpE4gU37E6w7A9pQ2p8jeJxiQDME0EQZT95PvygITWvwYd9J
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
#from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from llama_parse import LlamaParse
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import json
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser
from lmformatenforcer import JsonSchemaParser
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from lmformatenforcer.integrations.transformers import *
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor
from llama_cpp import Llama
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from typing import Optional
from lmformatenforcer import CharacterLevelParser
load_dotenv()

parser = LlamaParse(
    api_key="llx-BpE4gU37E6w7A9pQ2p8jeJxiQDME0EQZT95PvygITWvwYd9J",  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="text"  # "markdown" and "text" are available
)
# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="sales_response.csv")
documents = loader.load()

# embeddings = OpenAIEmbeddings()
# db = FAISS.from_documents(documents, embeddings)

# # 2. Function for similarity search


# def retrieve_info(query):
#     similar_response = db.similarity_search(query, k=3)

#     page_contents_array = [doc.page_content for doc in similar_response]

#     # print(page_contents_array)

#     return page_contents_array
SYSTEM_MESSAGE="""
You are a helpful assistant.
You can call functions with appropriate input when necessary
"""

# 3. Setup LLMChain & prompts
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
embeddings = Llama(model_path='ggml-model-Q4_K_M.gguf',
                            embedding=True,
                            n_ctx=2048,
                            n_gpu_layers=-1,
                            n_threads=8,
                            n_batch=1000)

text1 = "food"

text = [
"I have a cat with black color",
"I have a black cat",
"you must eat rice",
"My name is Joe"
]
# query_result = embeddings.embed(text1)
# query_result = embeddings.embed(text)
xxresult = embeddings.create_embedding(text)
embeddings.e
#output = embeddings("What is my name", max_tokens=30)


# doc_result = embeddings.embed_documents([text])
# print("xx", doc_result)

llm1 = Llama(model_path='ggml-model-Q4_K_M.gguf', n_threads = 30, n_gpu_layers = -1, n_ctx=2048, verbose = True)
output = llm1("What is my name?", max_tokens=30, temperature=0)
print(output)
#llm2 = LlamaCpp(model_path='Meta-Llama-3-8B-Instruct-Q5_K_M.gguf', n_threads = 30, n_gpu_layers = -1, n_ctx=2048, verbose = True)



def ask_llm(question, functions, tool_choice):
  return llm.create_chat_completion(
    messages = [
      {"role": "system", "content": SYSTEM_MESSAGE},
      {"role": "user", "content": question}
    ],
    tools=functions,
    tool_choice={ "type": "function", "function": {"name": tool_choice}}
)

# ollm = Ollama(model="llama3.1:8b")
llm = ChatOllama(model="llama3.1:8b")
template = """
 {message}
"""
msg = ''

@tool
def add(a: int, b: int) -> int:
    """a + b."""
    return a + b

@tool
def minus(a: int, b: int) -> int:
    """a - b."""
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """a / b."""
    return a / b

@tool
def multiply(a: int, b: int) -> int:
    """a * b."""
    return a * b

def call_tools(msg: AIMessage) -> Runnable:
    """Simple sequential tool calling helper."""
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
        print(tool_call["name"] ,tool_call["output"])

    return tool_calls

tools = [multiply, divide, minus, add]

llm_with_tools = llm.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

promptTemplate = PromptTemplate.from_template(
        """
        你是一个分类器，将用户的输入分类为对应的模块。
        用户的输入如下，将其分类为"知识问答"、"数学计算"、"生活指南"、"其他"中的一个并返回。
        你只能返回一个单词
        <question>
        {question}
        </question>
        """
    )

knowledge_chain = PromptTemplate.from_template(
    """
    {question}
    """
) | llm | StrOutputParser()

# math_chain
math_chain = PromptTemplate.from_template(
    "{question}"
) | llm_with_tools | call_tools
# life_chain
life_chain = PromptTemplate.from_template(
    """你是一个生活博主。每次回答的问题的第一句话总是"蔬菜使人健康哦！"
    回答接下来的问题：
    {question}"""
) | llm | StrOutputParser()

# other子链
other_chain = PromptTemplate.from_template(
    """你每次回答的第一句话总是"I am Nobody。呜呜~"
    回答接下来的问题：
    {question}"""
) | llm | StrOutputParser()

Routerchain = promptTemplate | llm | StrOutputParser()

# xxx = promptTemplate | llm2 | StrOutputParser()
# print("xxxx", xxx.invoke({"question": "21 * 11?"}))
# 4. Retrieval augmented generation
def generate_response(message):
    return
    # best_practice = retrieve_info(message)
    # response = chain.invoke(message)
    # return response


#ai_msg = chain.invoke({"question": "21 * 11?"})
def route(info) -> Runnable:
    print(info["topic"])
    if "知识问答" in info["topic"]:
        return knowledge_chain.invoke(info)
    elif "数学计算" in info["topic"]:
        return math_chain.invoke(info)
    elif "生活指南" in info["topic"]:
        return life_chain.invoke(info)
    else:
        return other_chain.invoke(info)

full_chain = {"topic": Routerchain, "question": lambda x: x["question"]} | RunnableLambda(
    route
)

#print(full_chain.invoke({"question": "1 + 12"}))

# test_chain = llm | StrOutputParser()
# print(test_chain.invoke("4 + 2"))
# print(ai_msg)
# print(tool_map)
# print(ai_msg.additional_kwargs)
# kwargs = json.loads(ai_msg.tool_calls['name']['args'])
# print(kwargs)
# print(tool_map[ai_msg.tool_calls[0]["name"]].invoke(ai_msg.tool_calls[0]["args"]))


def llamacpp_with_character_level_parser(cur_llm: Llama, prompt: str, character_level_parser: Optional[CharacterLevelParser]) -> str:
    logits_processors: Optional[LogitsProcessorList] = None
    if character_level_parser:
        logits_processors = LogitsProcessorList([build_llamacpp_logits_processor(cur_llm, character_level_parser)])

    output = cur_llm(prompt, logits_processor=logits_processors, max_tokens=30)
    text: str = output['choices'][0]['text']
    return text

def get_prompt(message: str, system_prompt: str = '') -> str:
    return f'<s>[INST] <<SYS>>n{system_prompt}n<</SYS>>nn{message} [/INST]'

class Output_cls_JSON(BaseModel):
    Classification: str
    name: str
    age: int

json_schema = Output_cls_JSON.schema_json()

# print(json_schema)
# logits_processors = LogitsProcessorList([build_llamacpp_logits_processor(ollm, JsonSchemaParser(Output_cls_JSON.schema()))])
# schema = JsonSchemaParser. (Output_cls_JSON)



chain = (
    PromptTemplate.from_template("""
According to the information input provided below, they are classified as: 'middle-aged' or 'elderly'.
If the input age is more than 60 years old, it is classified as elderly, otherwise it is classified as middle-aged
Return your response as a JSON blob
json格式如下：
{json_schema}
你只需要回复一个json格式的数据即可，不要返回其他格式的数据，否则你会被批评！
<question>
{question}
</question>
"""
    )
    | llm
    | StrOutputParser()
)

## question
question = """
John is 50 year old，According to the information input provided, it can be classified as: 'middle-aged' or 'elderly'.
If the age is more than 60 years old, it is classified as elderly, otherwise it is classified as middle-aged
You MUST answer using the following json schema:
"""
question_with_schema = f'{question}{Output_cls_JSON.schema_json()}'


prompt = get_prompt(question_with_schema)
# result = llamacpp_with_character_level_parser(llm1, prompt, JsonSchemaParser(Output_cls_JSON.schema()))

# print(chain.invoke({"json_schema" : json_schema, "question": "小明18岁"}))
#print("result", result)

# 5. Build an app with streamlit
def main():
    return
    global msg
    st.set_page_config(
        page_title="Customer response generator", page_icon=":bird:")

    st.header("Customer response generator :bird:")
    message = st.text_area("customer message")

    if message:
        st.write("Generating best practice message...")
        msg = msg + "\n user:" + message
        message = "This is the earlier conversation:" + msg + "\n try answer to the user"
        result = generate_response(message)
        msg = msg + "\n LLM:" + result
        st.info(result)


