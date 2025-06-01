# from formatted_tool import FormattedTool
# from langchain.agents import Tool
# from langchain.schema import AgentAction, AgentFinish
from typing import Annotated
from tool.generate_gbnf import EntityType
from NPC.baseNPC import BaseNPC
import inspect
import typing

# 工具函数
def move(self:BaseNPC, location:Annotated[str, EntityType.LOCATION]):
    location_obj = self.world_model.get_location_by_name(location)
    return self.arrive_location(location_obj)

def move_filter(self:BaseNPC) -> str:
    calls = []
    for location in self.location.adjacent_locations:
        calls.append(f"move('{location.name}'), ")
    if not calls:
        return ""
    return " ".join(calls)

def communicate(self:BaseNPC, npc:Annotated[str, EntityType.PERSON]):
    npc_obj = self.world_model.get_npc_by_name(npc)
    return self.communicate(npc_obj)

def communicate_filter(self:BaseNPC) -> str:
    calls = []
    for npc in self.location.npcs:
        if npc != self and npc.has_new_information:
            calls.append(f"communicate('{npc.name}'), ")
    
    if not calls:
        return ""
    return " ".join(calls)

def attack(self:BaseNPC, target:Annotated[str, EntityType.PERSON]):
    npc = self.world_model.get_npc_by_name(target)
    return self.battle(npc)

def attack_filter(self:BaseNPC) -> str:
    calls = []
    for npc in self.location.npcs:
        if npc != self and npc.dead == False:
            calls.append(f"attack('{npc.name}'), ")
    if not calls:
        return ""
    return " ".join(calls)

def use_inventory(self:BaseNPC, inventory:Annotated[str, EntityType.INVENTORY]):
    inventory = self.world_model.get_inventory_by_name(inventory)
    return self.use_inventory(inventory)

def use_inventory_filter(self:BaseNPC) -> str:
    calls = []
    for inventory in self.inventories:
        calls.append(f"use_inventory('{inventory.name}'), ")

    if not calls:
        return ""
    return " ".join(calls)

def buy_inventory(self:BaseNPC, inventory: Annotated[str, EntityType.INVENTORY], merchant: Annotated[str, EntityType.PERSON]):
    npc = self.world_model.get_npc_by_name(merchant)
    return self.buy_inventory(inventory, npc)

def buy_inventory_filter(self:BaseNPC) -> str:
    calls = []
    for npc in self.location.npcs:
        if npc != self:
            for inventory in npc.inventories:
                calls.append(f"buy_inventory('{inventory.name}', '{npc.name}'), ")
    return " ".join(calls)

# 生成函数签名和类型提示
def function_signature_with_types(func) -> str:
    sig = inspect.signature(func)
    type_hints = typing.get_type_hints(func)

    params_str = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue  # Skip self
        param_type = type_hints.get(name, "Any")
        if hasattr(param_type, '__name__'):
            param_type_str = param_type.__name__
        else:
            param_type_str = str(param_type)
        params_str.append(f"{name}: {param_type_str}")

    return_type = type_hints.get("return", "Any")
    if hasattr(return_type, '__name__'):
        return_type_str = return_type.__name__
    else:
        return_type_str = str(return_type)

    return f"def {func.__name__}({', '.join(params_str)}) -> {return_type_str}"

class ToolFunction:
    def __init__(self, func, filter_func, description: str = ""):
        self.name = func.__name__
        self.func = func
        self.filter_func = filter_func
        self.description = (function_signature_with_types(func) + ": " +  description)

tool_move=ToolFunction(
                func=move,
                filter_func=move_filter,
                description=(function_signature_with_types(move) # + ":Move to a location. The parameters need to include the location name."
                ),
            )

tool_communicate=ToolFunction(
                func=communicate,
                filter_func=communicate_filter,
                description= ("" # + ":communicate with other npc to gain information. The parameters need to include the name of the npc."
                ),
            )

tool_attack=ToolFunction(
                func=attack,
                filter_func=attack_filter,
                description=("Attack the enemy, who has lower power level will die."
                ),
            )

tool_use_inventory=ToolFunction(
                func=use_inventory,
                filter_func=use_inventory_filter,
                description=(":Use inventory or Equip a Equipment."
                ),
            )

tool_buy_inventory=ToolFunction(
                func=buy_inventory,
                filter_func=buy_inventory_filter,
                description=("" # + ":buy inventory. The parameters need to include the name of the inventory and the name of merchant."
                ),
            )

