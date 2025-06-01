from utility import model_instances as mi
import os
from openai import OpenAI


if __name__ == "__main__":
    # llm = mi.get_openai_instance()
    # result= llm.invoke("""
    #    hi, I am a brave hero named Thorn.
    # """)
    client = OpenAI(
    # If environment variables are not configured, replace the following line with: api_key="sk-xxx",
    api_key="sk-ce94790e897047c4a6dc141865e87b53", 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-turbo", # This example uses qwen-plus. You can change the model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': """
                    You are an assistant, according to the situation that player encountered, choose the appropriate aciton player can take.

        This is the situation :

        I am a brave hero named Thorn.
        My ultimate goal: Defeat the Demon Lord who resides in the Dark Castle.
        Currently my power level is 5, and I can't beat the Demon Lord whose power level is 10.

        what should I do next?


        Here is what you know about the world:
        ['Demon Lord is a monster at Dark Castle. Combat power: 10.', 'Demon Lord lives in the Dark Castle.', 'Thorn is a hero at .', 'You can only defeat the enemy whose power level is lower than you.', 'The demon king is afraid of toxic.', 'You can improve your power level by equipping weapons.', 'The Dark Castle is located in the north.', 'Amy is a merchant at Market. Sells: sword, fire dagger, Armor.', 'Gorim is a merchant at Village. Sells: rice, meat.', 'Mike is a merchant at Market. Sells: potion, poison.']

        the followng is the actions user can take and thier description:
        {'attack': 'Attack the enemy. The parameters need to include the target name. the definition: attack(target: str)', 'buy_item': 'buy item. The parameters need to include the name of the item and the name of merchant. the definition: buy_item(item: str, merchant: str)'}

        which tool is most appropriate action the for the situation? and give me the parameters, that is how you use the tool.
        The choice of parameters should allow the player to react correctly, in accordance with the will of the player.

        and tell me the reason you chose the action. the reason should be concise and clear within 1-2 sentences.
             """}],
        )
    
    response = completion.choices[0].message.content
    print(completion.model_dump_json())