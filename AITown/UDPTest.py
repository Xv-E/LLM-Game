import socket
import json
from utility import model_instances as mi
from agent.equipment_maker import WeaponGenerator
from llama_cpp import LlamaGrammar
# 服务器配置
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

weapon_grammar_path="./AITown/grammar/action_grammar.gbnf"
print(f"Python Server listening on {UDP_IP}:{UDP_PORT}")

# 处理 LLM 生成
def call_llm(prompt):
    llm = mi.get_llama_instance()

    generator = WeaponGenerator(llm)

    materials = [prompt]

    # 生成武器示例
    print("Generating a sword...")
    sword = generator.generate_weapon(materials)
    return sword
    # try:
    #     print("call_llm")
    #     response= mi.get_llama_instance().invoke(prompt, max_tokens=20)#, grammar=LlamaGrammar.from_file(weapon_grammar_path))
    #     print("call_llm", response)
    #     return response
    # except Exception as e:
    #     return f"LLM Error: {str(e)}"

# 处理存储数据
game_state = {}

def store_data(key, value):
    game_state[key] = value
    return f"Stored {key}: {value}"

def get_data(key):
    return game_state.get(key, "Key not found")

# UDP 监听
while True:
    data, addr = sock.recvfrom(4096)
    try:
        request = json.loads(data.decode())  # 解析 JSON
        action = request.get("action")

        if action == "llm":
            result = call_llm(request["prompt"])
        elif action == "store":
            result = store_data(request["key"], request["value"])
        elif action == "get":
            result = get_data(request["key"])
        else:
            result = "Invalid action"

        response = json.dumps({"result": result})
    except Exception as e:
        response = json.dumps({"error": str(e)})

    sock.sendto(response.encode(), addr)  # 发送 JSON 响应