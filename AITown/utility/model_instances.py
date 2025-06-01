from langchain_community.llms import LlamaCpp
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
from utility import embeddings
import atexit
import os
#sk-ce94790e897047c4a6dc141865e87b53
default_model_path='models/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf'
default_embedding_model_path='models/all-MiniLM-L6-v2'

_llama_instance = None
_openai_instance = None
_embedding_instance = None

def get_openai_instance(model_path=default_model_path, **kwargs):
    """获取全局 OpenAI 实例"""
    global _openai_instance
    if _openai_instance is None:
        _openai_instance = OpenAI(
            api_key= "sk-ce94790e897047c4a6dc141865e87b53", 
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    return _openai_instance

def get_ollama_instance(model_path=default_model_path, **kwargs):
    """获取全局 Ollama 实例"""
    global _llama_instance
    if _llama_instance is None:
        _llama_instance = Ollama(
            model="deepseek-r1:14b",
        )
    return _llama_instance

def get_llama_instance(model_path=default_model_path, **kwargs):
    """获取全局 LlamaCpp 实例"""
    global _llama_instance
    if _llama_instance is None:
        _llama_instance = LlamaCpp(
            model_path=model_path,
            temperature=0.5, #kwargs.get("temperature"),
            n_threads=12, #kwargs.get("n_threads"),
            n_gpu_layers=20, #kwargs.get("n_gpu_layers"),
            n_ctx=131072, #kwargs.get("n_ctx"),
            n_batch=256,
            verbose = False,
            #stop=["\nHuman:", "\nAI:", "\nhuman:", "\nai:"],
        )
    return _llama_instance

def get_embedding_instance(embedding_model_path=default_embedding_model_path):
    """获取全局嵌入模型实例"""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = embeddings.SentenceTransformerEmbeddings(path=embedding_model_path)
    return _embedding_instance

def destroy_singleton():
    """销毁单例实例"""
    global _llama_instance
    if _llama_instance is not None:
        _llama_instance.client.close()
        _llama_instance.client._sampler.close()
        _llama_instance = None

    global _embedding_instance
    if _embedding_instance is not None:
        #_embedding_instance.destroy()
        _embedding_instance = None
atexit.register(destroy_singleton)
