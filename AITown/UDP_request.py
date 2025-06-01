import socket
import json

# 服务器配置
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# 创建 UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 发送 LLM 请求
def send_llm_request(prompt):
    request = {
        "action": "llm",
        "prompt": prompt
    }
    
    # 将请求转换为 JSON 格式
    request_json = json.dumps(request)
    
    # 发送请求到服务器
    sock.sendto(request_json.encode(), (UDP_IP, UDP_PORT))
    
    # 接收响应
    data, addr = sock.recvfrom(4096)  # 最大接收 4096 字节
    response = json.loads(data.decode())
    
    return response

if __name__ == "__main__":
    # 你可以替换这个 prompt 为任何你想要发送给 LLM 的文本
    prompt = "What is the capital of France?"

    print(f"Sending LLM request with prompt: {prompt}")
    
    # 发送请求并打印响应
    response = send_llm_request(prompt)
    print("Received response:", response)
