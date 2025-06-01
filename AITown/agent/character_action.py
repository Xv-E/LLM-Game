from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import tool
from llama_cpp import LlamaGrammar

import sys
import os
# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from utility import prompt_template as pt
from utility import model_instances as mi
from tool import character_tools
from tool.grammar_tool_chain import ToolChain

# action_grammar_path = "./AITown/grammar/action_grammar.gbnf"

class CharacterAction:
    def __init__(self, llm):
        self.llm=llm
        #self.grammar = LlamaGrammar.from_file(action_grammar_path)
        
        tools=[
            character_tools.tool_move,
            character_tools.tool_pick_up,
            character_tools.tool_attack,
            character_tools.tool_use_item,
        ]

        self.toolchain = ToolChain(llm, tools)
        # 初始化代理
        # self.agent = initialize_agent(
        #     tools=character_tools.tools,
        #     llm=self.llm,
        #     handle_parsing_errors = True,
        #     max_iterations=3,
        #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # 使用基于描述的零次反应代理
        #     verbose=True
        # )

    def perceive_environment(self, environment):
        """感知环境并生成描述"""
        return f"Current environment information: {environment}"

    def generate_response(self, query, environment):
        """基于查询和环境生成响应"""
        # context = self.perceive_environment(environment)
        response = self.toolchain.invoke(query)
        return response

if __name__ == "__main__":
    # 初始化代理
    alice = CharacterAction(
        mi.get_llama_instance()
    )

    environment = {
    "position": (3, 3),
    "visible_tiles": [
        #{"position": (3, 4), "type": "item", "content": "gold_coin"},
        {"position": (4, 3), "type": "enemy", "content": {"name": "goblin", "hp": 10}},
        ]
    }
    query = "what action will you take?"
    response = alice.generate_response(query, environment)
    print(response)
