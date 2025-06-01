from typing import List, Dict, Callable
import sys
import os
# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from agent.worldModel import WorldModel

class BaseNPC:
    def __init__(self, name: str, location: str, world_model: WorldModel):
        self.name = name
        self.world_model = world_model
        self.inventories = []
        self.information = []
        self.has_new_information = True
        self.dead = False
        self.power_level = 0
        self.coins = 0
        self.location = world_model.get_location_by_name(location)
        world_model.register_entity(self)
        
    def describe(self):
        return f"{self.name} at {self.location.name} with {self.inventory}"
    
    def get_information(self) ->  List[str]:
        return self.information
    
    def add_information(self, information: str):
        self.information.append(information)
        self.has_new_information = True

    def battle(self, npc) -> bool:
        print(f"Battle: {npc.name}({npc.power_level}) vs {self.name} ({self.power_level})")
        if self.power_level > npc.power_level:
            npc.dead = True
            return True
        else:
            self.dead = True
            return False

    def sell_inventory(self, inventory_name: str, buyer):
        for inventory in self.inventories:
            if inventory.name == inventory_name:
                self.inventories.remove(inventory)
                buyer.inventories.append(inventory)
                print(f"{self.name} sell {inventory} to {buyer.name}.")
                break

    def buy_inventory(self, inventory_name: str, seller):
        for inventory in seller.inventories:
            if inventory.name == inventory_name:
                seller.inventories.remove(inventory)
                self.inventories.append(inventory)
                print(f"{self.name} buy {inventory} from {seller.name}.")

    def gain_inventory(self, inventory):
        self.inventories.append(inventory)
        print(f"{self.name} gain {inventory}.")


    def use_inventory(self, inventory):
        if inventory in self.inventories:
            self.inventories.remove(inventory)
            self.inventory.apply_to(self)
            return f"{self.name} use {inventory}.", True
        else:
            return f"{self.name} does not have {inventory}." , False

    # def get_location_obj(self):
    #     return self.world_model.get_location_by_name(self.location)

