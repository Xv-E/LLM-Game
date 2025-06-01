import inspect
import typing
from enum import Enum
from typing import Annotated

class EntityType(Enum):
    PERSON = "PERSON"
    INVENTORY = "INVENTORY"
    LOCATION = "LOCATION"

terms = {
    EntityType.PERSON: [''],
    EntityType.INVENTORY: [''],
    EntityType.LOCATION: ['']
}

def add_terms(entity_type: EntityType, term : str):
    """添加实体类型的术语到全局字典中"""
    if entity_type in terms and term not in terms[entity_type]:
        terms[entity_type].append(term)

def term_exist(entity_type: EntityType, term : str) -> bool:
    """检查实体类型的术语是否存在于全局字典中"""
    if entity_type in terms and term in terms[entity_type]:
        return True
    return False

def generate_gbnf_for_parameters(func):
    signature = inspect.signature(func)
    params = signature.parameters

    field_rules = []
    for name, param in params.items():
        if name == "self":
            continue 
        # 默认参数类型为 number
        param_type = "number"
        if typing.get_origin(param.annotation) is Annotated:
            base_type, *metadata = typing.get_args(param.annotation)
            if metadata and isinstance(metadata[0], EntityType):
                param_type = metadata[0].name
                #grammar_line = f'ws "\\"{name}\\":" ws ({options})'
        elif param.annotation == bool:
            param_type = "boolean"
        field_rules.append(f'ws "\\"{name}\\":" ws {param_type}')

    fields_str = " \",\" ".join(field_rules)

    annotation_type = '\n'.join(
        f"{entity_type.name} ::= {' | '.join( f'"\\"{name}\\""' for name in terms[entity_type])}"
        for entity_type in EntityType
        #if terms.get(entity_type) and len(terms[entity_type]) > 0
    )

    grammar = f"""
        root ::= format
        format ::= "{{" {fields_str} "}}"
        formatlist ::= "[]" | "[" ws format ("," ws format)* "]"
        string ::= "\\""   ([^\"\\n\\r{{}}]{{0,400}})   "\\""
        boolean ::= "true" | "false"
        ws ::= [ \\t\\n]?
        number ::= [0-9]+ "."? [0-9]*
        stringlist ::= "[" ws "]" | "[" ws string ("," ws string)* ws "]"
        numberlist ::= "[" ws "]" | "[" ws number ("," ws number)* ws "]"
        {annotation_type}
    """

    return grammar

def generate_gbnf_for_tools(tools):
    # Join tool definitions for tool_name rules
    tool_objects = " | ".join(
        f'"\\"{tool.name}\\""'
        for tool in tools
    )
    
    # GBNF definition
    grammar = f"""
        root ::= action
        type ::= {tool_objects}
        action ::= "{{"   ws   "\\"action_type\\":"   ws   type   ","   ws   "\\"parameters\\":"   ws   string ","   ws   "\\"reason\\":"   ws   string   "}}"
        actionlist ::= "[]" | "["   ws   action   (","   ws   action)*   "]"
        string ::= "\\""   ([^\"\\n\\r{{}}]{{0,200}})   "\\""
        boolean ::= "true" | "false"
        ws ::= [ \\t\\n]?
        number ::= [0-9]+   "."?   [0-9]*
        stringlist ::= "["   ws   "]" | "["   ws   string   (","   ws   string)*   ws   "]"
        numberlist ::= "["   ws   "]" | "["   ws   string   (","   ws   number)*   ws   "]"
    """
    return grammar

def generate_gbnf_for_func_qwen(func):
    signature = inspect.signature(func)
    params = signature.parameters
    field_rules = []
    for name, param in params.items():
        if name == "self":
            continue 
        # 默认参数类型为 number
        param_type = "number"
        if typing.get_origin(param.annotation) is Annotated:
            base_type, *metadata = typing.get_args(param.annotation)
            if metadata and isinstance(metadata[0], EntityType):
                param_type = metadata[0].name
                #grammar_line = f'ws "\\"{name}\\":" ws ({options})'
        elif param.annotation == bool:
            param_type = "boolean"
        field_rules.append(f'ws "\\"{name}\\":" ws {param_type}')

    fields_str = " \",\" ".join(field_rules)

    annotation_type = '\n'.join(
        f"{entity_type.name} ::= {' | '.join( f'"\\"{name}\\""' for name in terms[entity_type])}"
        for entity_type in EntityType
        #if terms.get(entity_type) and len(terms[entity_type]) > 0
    )

    grammar = f"""
        root ::= format
        format ::= "{{" {fields_str} ","   ws   "\\"prediction\\":"   ws   string   "}}"
        formatlist ::= "[]" | "[" ws format ("," ws format)* "]"
        string ::= "\\""   ([^\"\\n\\r{{}}]{{0,400}})   "\\""
        boolean ::= "true" | "false"
        ws ::= [ \\t\\n]?
        number ::= [0-9]+ "."? [0-9]*
        stringlist ::= "[" ws "]" | "[" ws string ("," ws string)* ws "]"
        numberlist ::= "[" ws "]" | "[" ws number ("," ws number)* ws "]"
        {annotation_type}
    """

    return grammar