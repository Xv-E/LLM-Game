from typing import Dict, Any, List, Annotated
from langchain.schema.runnable import RunnableSequence, RunnableLambda
from utility import prompt_template as pt
from llama_cpp import LlamaGrammar
import json
import inspect
import typing
from tool.generate_gbnf import generate_gbnf_for_parameters, generate_gbnf_for_tools
from langchain.agents import Tool

class ToolChain:
    tools: List[Tool] = []
    func_grammar: LlamaGrammar = None
    parameter_grammars: Dict[str, LlamaGrammar] = {}
    tool_descriptions: Dict[str, str] = {}

    ### pipeline func end
    def __init__(self, owner, llm, tools:List[Tool]):
        self.llm = llm
        self.tools=tools
        self.owner=owner
        # generate gbnf for tools
        cur_gbnf_str = generate_gbnf_for_tools(tools)
        self.func_grammar=LlamaGrammar.from_string(cur_gbnf_str)
        # print('select_tool cur_grammar ', cur_gbnf_str)
        
        # generate gbnf for parameters
        self.parameter_grammars={}
        # terms更新需重新生成
        # for tool in tools:
        #     cur_gbnf_str = generate_gbnf_for_parameters(tool.func)
        #     cur_grammar=LlamaGrammar.from_string(cur_gbnf_str)
        #     self.parameter_grammars[tool.name]=cur_grammar
        
        self.tool_descriptions={}
        for tool in tools:
            self.tool_descriptions[tool.name]=tool.description

        debug_step = RunnableLambda(
            func=lambda inputs: (print(f"Debug: {inputs}"), inputs)[1]
        )
        select_func_step=RunnableLambda(self.select_tool)

        parse_parameter_step=RunnableLambda(
             func=lambda inputs: (
                lambda parsed: {
                    "tool": inputs["tool"],
                    "parameters": parsed,
                    # "reason": parsed[1]
                }
            )(self.parse_parameters(inputs))
        )
       
        call_func_step=RunnableLambda(
            func=lambda inputs: {
                "result": inputs["tool"].func(self.owner, **inputs["parameters"])
            }
        )

        # 创建 RunnableSequence
        self.pipeline = select_func_step | debug_step| parse_parameter_step | debug_step| call_func_step

    # 定义工具选择逻辑
    def select_tool(self, inputs: str) -> str:
        input_text = inputs["input_text"]
        information = self.owner.world_model.query_facts(input_text)
        cur_prompt = pt.select_action_template.format(
            input=input_text,
            tools=self.tool_descriptions,
            information=information
        )
        print("select_tool_prompt: ", cur_prompt)
        cur_grammar=self.func_grammar
        result = self.llm.invoke(cur_prompt, grammar=cur_grammar)
        # print(result)
        try:
            python_obj = json.loads(result)
        except json.JSONDecodeError as e:
            print("❌ JSON Decode Error:")
            print(f"Error: {e}")
            print("🔍 Raw result:")
            print(result)
            raise e
        #python_obj = json.loads(result)
        
        tool_name = python_obj["action_type"]
        reason = python_obj['reason']
        params = python_obj["parameters"]
        chosen_tool = None
        for tool in self.tools:
            if tool.name == tool_name:
                chosen_tool = tool
                break

        if tool == None:
            raise Exception(f"Invalid tool name {tool_name}")
       
        return {
            "tool": chosen_tool,
            "reason": reason,
            "parameters": params,
            "original_text": input_text,
        }
    
    # 参数解析逻辑
    def parse_parameters(self, input) -> Dict[str, Any]:
        #tool: Tool, reason: str, input_text: str
        tool = input["tool"]
        reason = input["reason"]
        input_text = input["original_text"]
        parameters = input["parameters"]
        signature = inspect.signature(tool.func)
        # parameters = ""

        # for name, param in signature.parameters.items():
        #     if name == "self":
        #         continue
        #     parameters+=f"{name},"
        # parameters = parameters[:-1]  # 去掉最后一个逗号

        infomation = self.owner.world_model.query_facts(reason)

        cur_prompt = pt.parameter_parse_template.format(
            input=input_text,
            tool=tool.name + ':' + self.tool_descriptions[tool.name],
            reason=reason,
            parameters=parameters,
            information=infomation
        )
        print("parse_parameters cur_prompt: ", cur_prompt)
        #cur_grammar=self.parameter_grammars[tool.name]
        # 每次重新生成参数的gbnf
        cur_gbnf_str = generate_gbnf_for_parameters(tool.func)
        cur_grammar=LlamaGrammar.from_string(cur_gbnf_str)
        result = self.llm.invoke(cur_prompt, grammar=cur_grammar)
        try:
            python_obj = json.loads(result)
            return python_obj
        except json.JSONDecodeError as e:
            print("❌ JSON Decode Error:")
            print(f"Error: {e}")
            print("🔍 Raw result:")
            print(result)
            raise e
    
    def invoke(self, user_input):
        result = self.pipeline.invoke({"input_text": user_input})
        return result
