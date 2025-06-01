from llama_cpp import LlamaGrammar
from langchain.schema.runnable import RunnableSequence, RunnableLambda
import json
import sys
import os
# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from utility import prompt_template as pt
from utility import model_instances as mi

weapon_grammar_path="./AITown/grammar/weapon_grammar.gbnf"


class WeaponGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.template = pt.weapon_template
        self.grammar = LlamaGrammar.from_file(weapon_grammar_path)

    def generate_weapon(self, materials):
        """Generate a weapon with optional constraints."""

        cur_prompt = pt.weapon_template.format(
            materials=materials,
        )
        print("cur_prompt", cur_prompt)
        
        response = self.llm.invoke(cur_prompt, grammar=self.grammar)
        #weapon = json.loads(response)
        return response

# 示例使用
if __name__ == "__main__":
    llm = mi.get_llama_instance()

    generator = WeaponGenerator(llm)

    materials = ["Iron nail", "Wooden stick"]

    # 生成武器示例
    print("Generating a sword...")
    sword = generator.generate_weapon(materials)
    print(json.dumps(sword, indent=4))

