from typing import List, Dict, Callable
import sys
import os
# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from NPC.baseNPC import BaseNPC
from agent.worldModel import WorldModel

class MerchantNPC(BaseNPC):
    def __init__(self, world_model: WorldModel, name: str, location: str, inventories: List[str]): 
        super().__init__(name, location, world_model)
        for inventory in inventories:
            inventory = world_model.get_inventory_by_name(inventory)
            if inventory is None:
                raise ValueError(f"Inventory {inventory} not found in world model.")
            self.inventories.append(inventory)
          
    def describe(self) -> str:
        """描述商人的信息"""
        inventories = [inventory.name for inventory in self.inventories]
        if not inventories:
            inventories = ["nothing"]
        return f"{self.name} is a merchant at {self.location.name} #Sells: {', '.join(inventories)}."

    def get_keywords(self) -> List[str]:
        return ["merchant", "shop", "buy", "sell", self.name] 
