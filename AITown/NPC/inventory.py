from typing import List

class Inventory:
    def __init__(self, world, name, description):
        self.name = name
        self.description = description
        self.information = []
        self.effect = None
        self.world = world
        self.world.register_inventory(self)

    def describe(self) -> str:
        return f"{self.name}: {self.description}"
    
    def get_information(self) ->  List[str]:
        return self.information
    
    def add_information(self, information: str):
        self.information.append(information)

    def add_effect(self, effect):
        self.effect = effect

    def apply_to(self, user):
        """应用物品效果到 user 对象"""
        if self.effect:
            self.effect(user)  # 调用函数指针
            return f"{user.name} used {self.name}."
        else:
            return f"{self.name} has no effect."
    
    def __str__(self):
        return self.name