from typing import Dict, Any, List, Annotated
from langchain.schema.runnable import RunnableSequence, RunnableLambda
from utility import model_instances as mi
from utility import prompt_template as pt
from llama_cpp import LlamaGrammar
import json
import inspect
import typing
from tool.generate_gbnf import generate_gbnf_for_parameters, generate_gbnf_for_tools
from tool.character_tools import ToolFunction

class ToolChain_Qwen:
    tools: List[ToolFunction] = []
    func_grammar: LlamaGrammar = None
    parameter_grammars: Dict[str, LlamaGrammar] = {}
    tool_descriptions: Dict[str, str] = {}

    ### pipeline func end
    def __init__(self, owner, llm, tools:List[ToolFunction]):
        self.llm = llm
        self.tools=tools
        self.owner=owner
        self.parse_log = ""
        self.round = 0
        # generate gbnf for tools
        cur_gbnf_str = generate_gbnf_for_tools(tools)
        self.func_grammar=LlamaGrammar.from_string(cur_gbnf_str)

        # generate gbnf for parameters
        #self.parameter_grammars={}
        # self.tool_descriptions={}
        # for tool in tools:
        #     self.tool_descriptions[tool.name]=tool.description

        debug_step = RunnableLambda(
            func=lambda inputs: (print(f"Debug: {inputs}"), inputs)[1]
        )
        select_func_step=RunnableLambda(self.select_tool)

        parse_parameter_step=RunnableLambda(
             func=lambda inputs: (
                lambda parsed: {
                    "tool": inputs["tool"],
                    "parameters": parsed
                }
            )(self.parse_parameters(inputs))
        )
       
        call_func_step=RunnableLambda(func=self.call_func)

        # 创建 RunnableSequence
        self.pipeline = select_func_step | debug_step| parse_parameter_step | debug_step| call_func_step

    def fallback_tool(self) -> str:
        return "there is no matching action", False

    def generate_available_tools_prompt(self) -> str:
        """生成可用工具列表（仅包含有可能调用的工具）"""
        available_tools = []
        for tool in self.tools:
            tool_call_str = tool.filter_func(self.owner)
            if tool_call_str != "":
                available_tools.append(f"{tool.name}: {tool.description} Possible Calls:")
                available_tools.append(tool_call_str)
        return "\n".join(available_tools)

    # 定义工具选择逻辑
    def select_tool(self, inputs: str) -> str:
        input_text = inputs["input_text"]
        status = self.owner.get_status()
        current_goal = self.owner.get_current_goal()
        current_reason = self.owner.goalManager.current_reason
        information = self.owner.memory.query_memory(current_goal)
        tools = self.generate_available_tools_prompt()

        cur_prompt = pt.select_action_template_qwen.format(
            status=status,
            current_goal=current_goal,
            reason=current_reason,
            tools=tools,
            information=information,
        )

        print("select_tool_prompt: ", cur_prompt)
        
        completion = self.llm.chat.completions.create(
            model="qwen-turbo-latest", # This example uses qwen-plus. You can change the model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': str(cur_prompt)}, 
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        python_obj = json.loads(content)
        print("content: ", content)
        tool_name = python_obj.get("action", "")
        params = python_obj.get("params", "")
        reason = python_obj.get("reason", "")
        
        chosen_tool = None
        for tool in self.tools:
            if tool.name == tool_name:
                chosen_tool = tool
                break
        
        # 如果没有找到匹配的工具，使用默认的工具
        if chosen_tool == None:
            chosen_tool = self.fallback_tool
            params = {}

        # use grammer
        if tool == None:
            print("tool is None, use grammar")
            result = mi.get_llama_instance().invoke(tool_name, grammar=self.func_grammar)
            python_obj = json.loads(result)
            tool_name = python_obj["action_type"]
            for tool in self.tools:
                if tool.name == tool_name:
                    chosen_tool = tool
                    break
                if tool == None:
                    raise Exception(f"Invalid tool name {tool_name}")
       
        return {
            "tool": chosen_tool,
            "parameters": params,
            "reason": reason,
        }
    
    # 参数解析逻辑
    def parse_parameters(self, input) -> Dict[str, Any]:
        #tool: Tool, reason: str, input_text: str
        tool = input["tool"]
        parameters_obj = input["parameters"]
        reason = input.pop("reason")
        self.parse_log = ""
        signature = inspect.signature(tool.func)
        params = signature.parameters

        need_parse = False
        unexpected_params = set(parameters_obj.keys()) - set(params.keys())
        if unexpected_params:
            self.parse_log = f"Unexpected parameters provided: {unexpected_params}"
            need_parse = True

        if need_parse == False:
            for name, param in params.items():
                if name == "self":
                    continue 

                if parameters_obj.get(name) is None:
                    self.parse_log = f"Parameter '{name}' is missing."
                    need_parse = True
                    break

                param_value = parameters_obj[name]
                if typing.get_origin(param.annotation) is Annotated:
                    base_type, *metadata = typing.get_args(param.annotation)
                    if metadata and not self.owner.world_model.term_exist(metadata[0], param_value):
                        self.parse_log = f"Parameter '{name}' with value '{param_value}' is not valid for type '{metadata[0].name}'."
                        need_parse = True
                        break
        
        if not need_parse:
            # 不需要解析参数，直接返回
            return parameters_obj
        
        print(self.parse_log)
        # parse parameters
        tool_description = ''
        for t in self.tools:
            if t.name == tool.name:
                tool_description = t.description
                break
        
        goal = self.owner.get_current_goal()
        cur_prompt = pt.parameter_parse_template_qwen.format(
            hero_name=self.owner.name,
            tool=tool_description,
            reason=reason,
            goal = goal,
            parameters=parameters_obj,
        )
        print("use grammar, parse_parameters cur_prompt: ", cur_prompt)
        #cur_grammar=self.parameter_grammars[tool.name]
        # 每次重新生成参数的gbnf
        cur_gbnf_str = generate_gbnf_for_parameters(tool.func)
        cur_grammar=LlamaGrammar.from_string(cur_gbnf_str)
        result = mi.get_llama_instance().invoke(cur_prompt, grammar=cur_grammar)
        try:    
            python_obj = json.loads(result)
            return python_obj
        except json.JSONDecodeError as e:
            print("❌ JSON Decode Error:")
            print(f"Error: {e}")
            print("🔍 Raw result:")
            print(result)
            raise e
    
    def call_func(self, inputs):
        tool = inputs["tool"]
        parameters = inputs["parameters"]
        result, flag = tool.func(self.owner, **parameters)

        status = "SUCCESS" if flag else "FAIL"
        # self.owner.action_recode.append(
        #     f"{tool.name}({', '.join([f'{k}={v}' for k, v in parameters.items()])}) [{status}] -> Result: {result}"
        # )
 
        self.owner.action_recode.append(
        f"""
        Round {self.round}:
        Action: {tool.name}({', '.join([f'{k}={v}' for k, v in parameters.items()])})
        Goal: {self.owner.get_current_goal()}
        Status: {"SUCCESS" if flag else "FAIL"}
        Result: {result}
        """
        )
        self.round += 1
        # 保持只有最近10条记录
        if len(self.owner.action_recode) > 10:
            self.owner.action_recode = self.owner.action_recode[-10:]
        return result, flag

    def update_goal(self, last_action_result: str, flag: bool):
        status = self.owner.get_status()
        environment = self.owner.location.describe()
        clue = self.owner.memory.get_task_clue('main')
        Parent_goal = self.owner.get_parent_goal()
        current_goal = self.owner.get_current_goal()
        recent_actions = chr(10).join(self.owner.action_recode)
        tools = self.generate_available_tools_prompt()
        cur_prompt = pt.update_goal_template.format(
            parent_goal=Parent_goal,
            current_goal=current_goal,
            environment=environment,
            status=status,
            tools=tools,
            clue=clue,
            recent_actions=recent_actions,
            last_action_result=last_action_result,
            flag = "success" if flag == True else "fail",
            current_goal_problem = f"\nthere is some problems during last action execution: {self.parse_log}" if self.parse_log != "" else "",
        )

        print("select_tool_prompt: ", cur_prompt)
        
        completion = self.llm.chat.completions.create(
            model="qwen-turbo-lastest", # This example uses qwen-plus. You can change the model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': str(cur_prompt)}, 
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        python_obj = json.loads(content)
        print("content: ", content)
        #achieved = python_obj["achieved"]
        new_sub_goal = python_obj['new_sub_goal']
        reason = python_obj["reason"]

        self.owner.goalManager.push_goal(new_sub_goal) 
        self.owner.goalManager.current_reason = reason
        return flag
        

    def invoke(self, user_input):
        result = self.pipeline.invoke({"input_text": user_input})
        return result
    
    def step(self, user_input):
        hero = self.owner

        result, flag = self.pipeline.invoke({"input_text": user_input})
        
        hero.action.update_goal(result, flag)
        return result
