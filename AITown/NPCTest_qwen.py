import socket
import json
from utility import model_instances as mi
# from agent.equipment_maker import WeaponGenerator
from llama_cpp import LlamaGrammar
from agent.worldModel import WorldModel
from NPC.location import Location
from NPC.hero import Hero
from NPC.merchantNPC import MerchantNPC
from NPC.monster import Monster
from NPC.inventory import Inventory 
from utility import model_instances as mi
from tool.generate_gbnf import EntityType

def main_loop(hero: Hero, world: WorldModel):
    print("\n🎮 游戏开始！英雄目标：打败 Demon Lord\n")

    while True:
        # 简单条件：如果 Demon Lord 被打败，游戏胜利
        demon_lord = world.get_npc_by_name("Demon Lord")
        if demon_lord and demon_lord.dead:
            print("\n🎉 恭喜！你打败了 Demon Lord！游戏胜利！")
            break

        # 每回合英雄尝试采取下一步动作
        result, flag = hero.take_action("Take the next best step to defeat the Demon Lord.")
        input("🔁 按 Enter 更新目标...\n")  # 暂停一回合
        hero.action.update_goal(result, flag)
        print(f"📍 Location：{hero.location.name}")
        print(f"🎒 inventory：{[str(item) for item in hero.inventories] if hero.inventories else 'none'}")
        print(f"💪 power level：{hero.power_level}")
        input("🔁 按 Enter 执行下一回合...\n")  # 暂停一回合

if __name__ == "__main__":
    embedding_model = mi.get_embedding_instance()
    #llm = mi.get_llama_instance()
    llm = mi.get_openai_instance()
    world = WorldModel(embedding_model)
    world.add_fact("Demon Lord lives in the Dark Castle.")
    world.add_fact("The Dark Castle is located in the north.")
    world.add_fact("You can improve your power level by equipping weapons.")
    world.add_fact("You can only defeat the enemy whose power level is lower than you.")

    # 注册地点
    Market = Location(world, "Market", "A bustling market filled with various goods.")
    Market.add_information("You can buy items from the merchants here.")

    Village = Location(world, "Village", "A peaceful village with friendly villagers.")
    Village.add_information("The villagers are friendly and may have useful information.")

    DarkCastle = Location(world, "Dark Castle", "A dark and eerie castle filled with monsters.")
    DarkCastle.add_information("The Demon Lord resides here.")

    Forest = Location(world, "Forest", "A dense forest with many hidden paths.")
    Forest.add_information("There are many wild animals in the forest.")

    Village.add_adjacent_locations([Market, DarkCastle, Forest])

    # 注册物品
    sword = Inventory(world, "Sword", "A sharp sword.  power_level + 2")
    sword.add_effect(lambda user: setattr(user, "power_level", user.power_level + 2))
    fire_dagger = Inventory(world, "Fire Dagger", "A dagger that burns with fire.  power_level + 4")
    fire_dagger.add_effect(lambda user: setattr(user, "power_level", user.power_level + 4))
    armor = Inventory(world, "Armor", "A sturdy armor.  power_level + 3")
    armor.add_effect(lambda user: setattr(user, "power_level", user.power_level + 3))

    rice = Inventory(world, "Rice", "A bowl of rice. power_level + 1")
    rice.add_effect(lambda user: setattr(user, "power_level", user.power_level + 1))
    meat = Inventory(world, "Meat", "A piece of meat. power_level + 2")
    meat.add_effect(lambda user: setattr(user, "power_level", user.power_level + 2))

    potion = Inventory(world, "Potion", " power_level + 1")
    potion.add_effect(lambda user: setattr(user, "power_level", user.power_level + 1))
    posion = Inventory(world, "Poison", "A deadly poison.  power_level - 2")
    posion.add_effect(lambda user: setattr(user, "power_level", user.power_level - 2))

    
    # 注册怪物
    demon_lord = Monster("Demon Lord", "Dark Castle", 10, world)

    # 注册商人
    Mike = MerchantNPC(world, "Mike", "Market", ["Potion", "Poison"])
    Mike.add_information("The demon king is afraid of flame.")

    Gorim = MerchantNPC(world, "Gorim", "Village", ["Rice", "Meat"])
    Gorim.add_information("The demon king lives in Dark Castle.")
    Gorim.add_information("The power level of demon king is 10.")

    Amy = MerchantNPC(world, "Amy", "Market", ["Sword", "Fire Dagger", "Armor"])
    Amy.add_information("The Dark Castle is located in the north.")  

    # 创建一个英雄
    hero = Hero(world,"Thorn", "Village", llm, embedding_model)
    hero.power_level = 5
    # hero.add_memory("You can improve your power level by equipping equipments.")
    # hero.add_memory("You can only defeat the enemy whose power level is lower than you.")
    # hero.add_memory("You can communicate with the people to gain more information. Multiple conversations won't bring any new information")
    # hero.add_memory("You can buy items from the merchant.")
    # hero.add_memory("You can use items only if you own it.")

    #hero.goalManager.push_goal("Defeat the Demon Lord")
    hero.goalManager.push_goal("Gain more information")
    # 英雄尝试实现目标

    # hero.take_action("""
    #     I am hungry.
    # """)
    # hero.action.update_goal("I can't defeat Demon Lord now.", False)
    input("🔁 按 Enter 执行下一回合...\n")  # 暂停一回合

    # print("玩家背包：", [str(item) for item in hero.inventories])
    main_loop(hero, world)