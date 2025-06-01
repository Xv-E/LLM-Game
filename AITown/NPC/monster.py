from typing import List, Dict, Callable
import sys
import os
# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from NPC.baseNPC import BaseNPC
from agent.worldModel import WorldModel

class Monster(BaseNPC):
    def __init__(self, name: str, location: str, power_level: int, world_model: WorldModel): 
        super().__init__(name, location, world_model)
        self.power_level = power_level
        
    def describe(self) -> str:
        return f"{self.name} is a monster at {self.location}. Combat power: {self.power_level}."

    def get_keywords(self) -> List[str]:
        return ["merchant", "shop", "buy", "sell"] + self.inventory
