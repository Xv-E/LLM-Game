from langchain_community.llms import LlamaCpp
import embeddings

_llama_instance = None
_embedding_instance = None

def get_llama_instance(model_path, **kwargs):
    """获取全局 LlamaCpp 实例"""
    global _llama_instance
    if _llama_instance is None:
        _llama_instance = LlamaCpp(
            model_path=model_path,
            temperature=kwargs.get("temperature", 0.3),
            n_threads=kwargs.get("n_threads", 30),
            n_gpu_layers=kwargs.get("n_gpu_layers", -1),
            n_ctx=kwargs.get("n_ctx", 2024),
            stop=["\nHuman:", "\nAI:", "\nhuman:", "\nai:"],
        )
    return _llama_instance

def get_embedding_instance(embedding_model_path):
    """获取全局嵌入模型实例"""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = embeddings.SentenceTransformerEmbeddings(path=embedding_model_path)
    return _embedding_instance