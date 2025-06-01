import socket
import json
from utility import model_instances as mi
# from agent.equipment_maker import WeaponGenerator
from llama_cpp import LlamaGrammar
from agent.worldModel import WorldModel
from NPC.hero import Hero
from NPC.merchantNPC import MerchantNPC
from NPC.monster import Monster
from utility import model_instances as mi
from tool.generate_gbnf import EntityType

if __name__ == "__main__":
    embedding_model = mi.get_embedding_instance()
    llm = mi.get_llama_instance()
    
    world = WorldModel(embedding_model)
    world.add_fact("Demon Lord lives in the Dark Castle.")
    world.add_fact("The Dark Castle is located in the north.")
    world.add_fact("The demon king is afraid of flames.")

    # 注册怪物
    monster = Monster("Demon Lord", "Dark Castle", 10, world)
    # 注册商人
    merchant1 = MerchantNPC("Mike", "Market", ["potion", "poison"], world)
    merchant2 = MerchantNPC("Gorim", "Village", ["rice", "meat"], world)
    merchant3 = MerchantNPC("Amy", "Market", ["sword", "fire dagger", "Armor"], world)
    # 创建一个英雄
    hero = Hero("Thorn", world, llm, embedding_model)
    hero.power_level = 5
    hero.location = "Village"
    # 英雄尝试实现目标
    hero.take_action("""
        I am a brave hero named Thorn.
        My ultimate goal: Defeat the Demon Lord who resides in the Dark Castle.
        Currently my power level is 5, and I can't beat the Demon Lord whose power level is 10.
        
        what should I do next?
    """)
    
    print(f"[INFO] 商人：{merchant1.name} | 背包：{', '.join(merchant1.inventory) if merchant1.inventory else '空'}")
    print(f"[INFO] 商人：{merchant2.name} | 背包：{', '.join(merchant2.inventory) if merchant2.inventory else '空'}")
    print(f"[INFO] 商人：{merchant3.name} | 背包：{', '.join(merchant3.inventory) if merchant3.inventory else '空'}")
    print("玩家背包：", hero.inventory)