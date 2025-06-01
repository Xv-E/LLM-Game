from typing import List, Dict
import sys
import os
#import numpy as np
# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from tool.grammar_tool_chain import ToolChain
from tool.grammar_tool_chain_qwen import ToolChain_Qwen
from tool import character_tools
from agent.memoryManager import MemoryManager
from agent.worldModel import WorldModel
from utility.embeddings import SentenceTransformerEmbeddings
from NPC.baseNPC import BaseNPC
from NPC.location import Location
from agent.worldModel import WorldModel
from NPC.goalManager import GoalManager
class Hero(BaseNPC):
    def __init__(self, world_model: WorldModel, name: str, location: str, llm, Embedding : SentenceTransformerEmbeddings):
        super().__init__(name, location, world_model)
        self.memory = MemoryManager(Embedding)
        self.equipment = []
        self.goalManager = GoalManager("hero", self.memory.query_memory)
        self.llm = llm
        self.action_recode = []
        tools=[
            character_tools.tool_move,
            character_tools.tool_attack,
            character_tools.tool_communicate,
            character_tools.tool_buy_inventory,
            character_tools.tool_use_inventory,
        ]

        self.action = ToolChain_Qwen(self, llm, tools)

    def describe(self) -> str:
        return f"{self.name} is a hero at {self.location}."

    def get_keywords(self) -> List[str]:
        return ["hero"]
    
    def communicate(self, npc: BaseNPC):
        if npc == self:
            return (f"{self.name} cannot communicate with itself."), False
        if npc.dead:
            return (f"{self.name} cannot communicate with a dead person."), False
        if npc.has_new_information == False:
            return (f"{self.name} has no new information to communicate with {npc.name}."), False
        
        for information in npc.get_information():
            self.memory.add_task_clue("main", information)
        
        self.memory.add_memory(npc.describe(), npc.get_keywords(), npc.name)
        npc.has_new_information = False
        return (f"{self.name} communicate with {npc.name}, and I know {npc.get_information()}"), True
    
    def arrive_location(self, location: Location):
        if location == self.location:
            return (f"{self.name} is already at {location.name}."), False
        if location not in self.location.adjacent_locations:
            return (f"{self.name} cannot arrive at {location.name}, because it is not adjacent."), False
        
        self.location = location
        for information in location.get_information():
                self.memory.add_memory(information)
        return (f"{self.name} arrive at {location.name}"), True

    def battle(self, npc: BaseNPC):
        print(f"Battle: {npc.name}({npc.power_level}) vs {self.name} ({self.power_level})")
        if self.power_level > npc.power_level:
            npc.dead = True
            return (f"Battle: {self.name}({self.power_level}) beat {npc.name} ({npc.power_level}), {npc.name} is dead."), True
        else:
            self.dead = True
            return (f"Battle: {self.name}({self.power_level}) lose to {npc.name} ({npc.power_level}), {self.name} is dead."), False

    def buy_inventory(self, inventory_name: str, seller):
        if seller not in self.location.npcs:
            return (f"{self.name} cannot buy {inventory_name} from {seller.name}, because {seller.name} is not in the same location."), False
        for inventory in seller.inventories:
            if inventory.name == inventory_name:
                seller.inventories.remove(inventory)
                self.inventories.append(inventory)
                return (f"{self.name} buy {inventory} from {seller.name}."), True
        return (f"{self.name} cannot buy {inventory_name} from {seller.name}, because {seller.name} does not have it."), False
    
    def use_inventory(self, inventory):
        if inventory in self.inventories:
            self.inventories.remove(inventory)
            inventory.apply_to(self)
            self.equipment.append(inventory.name)
            return (f"{self.name} use {inventory}。"), True
        else:
            return (f"{self.name} does not have {inventory}."), False

    def query_memory(self, query: str, k = 10) -> List[str]:
        return self.memory.query_memory(query, k)

    def get_status(self) -> str:
        """获取状态信息"""
        if self.inventories:
            inventories = [item.name for item in self.inventories]
        else:
            inventories = []
        return f"{self.name} status:\n location:{self.location.name}\n bag:{inventories}\n equipped:{self.equipment}\n power level:{self.power_level}"

    def get_current_goal(self) -> str:
        """获取当前目标"""
        return self.goalManager.get_current_goal()
    
    def get_parent_goal(self) -> str:
        """获取父目标"""
        return self.goalManager.get_parent_goal()

    def take_action(self, goal: str):
        return self.action.invoke(goal)
    
    def main_loop(self):
        while True:
            # 获取当前目标
            current_goal = self.get_current_goal()
            if current_goal is None:
                print("No current goal.")
                break
            
            # 更新目标
            self.goalManager.update_goal(result, True)
            # 执行当前目标
            result = self.take_action(current_goal)
            print(result)
            



