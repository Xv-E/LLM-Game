from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import LlamaCpp

# 使用 OpenAI 的聊天模型
#llm = ChatOpenAI(temperature=0)
llm = LlamaCpp(model_path='ggml-model-Q4_K_M.gguf', temperature=0.0, n_threads = 30, n_gpu_layers = -1, n_ctx=2024, stop=[["\nHuman:", "\nAI:"]])
# 配置内存
memory = ConversationBufferMemory()

# 创建对话链
conversation = ConversationChain(llm=llm, memory=memory)

# 开始对话
print(conversation.run("你好，我是小明"))
print(conversation.run("我之前说过我是谁吗？"))