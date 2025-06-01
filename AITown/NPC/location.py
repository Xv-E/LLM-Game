from typing import List

class Location:
    def __init__(self, world, name, description, npcs=None):
        self.name = name
        self.description = description
        self.npcs = npcs or []
        self.adjacent_locations = []
        self.information = []
        world.register_location(self)

    def add_npc(self, npc):
        """添加 NPC 到地点"""
        self.npcs.append(npc)

    def describe(self) -> str:
        npc_list = ", ".join([npc.name for npc in self.npcs]) if self.npcs else "no NPC"
        neighbor_list = ", ".join([loc.name for loc in self.adjacent_locations]) if self.adjacent_locations else "no adjacent locations"
        return (
            f"you are in {self.name}: {self.description}\n"
            f"there are people you can interact with:{npc_list}\n"
            f"You can go to:{neighbor_list}"
        )
    
    def get_information(self) ->  List[str]:
        return self.information
    
    def add_information(self, information: str):
        self.information.append(information)

    def add_adjacent_locations(self, locations: List["Location"],  bidirectional: bool = True):
        """添加相邻地点"""
        for loc in locations:
            if loc not in self.adjacent_locations:
                self.adjacent_locations.append(loc)

            if bidirectional and self not in loc.adjacent_locations:
                loc.adjacent_locations.append(self)