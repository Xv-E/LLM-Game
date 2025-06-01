from langchain.prompts import PromptTemplate

# 5. Create the prompt template using the custom template
character_PT = PromptTemplate(
    input_variables=["role_settings", "query", "memory"], 
    template="""
        You are a NPC in a game.
        
        This is your role setting:
        {role_settings}

        This is relevant momery about the user input:
        {memory}
        
        This is the user input:
        {query}

        now make your response, make it concise, Write your response directly, 
        without any paraphrasing or characteristic quotes, and without extra line breaks:
        """
    )

weapon_template = PromptTemplate(
    input_variables=["materials"], 
    template="""
        You are an expert weapon designer. 
        
        This weapon is made of the following materials:
        {materials}
        
        Generate a weapon which must conform to the characteristics of the material, 
        and make it in JSON format with the following attributes:
        - name: The name of the weapon.
        - type: Weapon type (e.g., "Sword", "Bow", "Staff").
        - rarity: Rarity of the weapon (Common, Rare, Epic, Legendary).
        - base_damage: Base damage value (integer).
        - element: Optional elemental type (e.g., "Fire", "Ice", "Lightning", or None).
        - description: A short description of the weapon.
        - skills: A list of special skills the weapon provides. Each skill has:
        - name: The name of the skill.
        - cooldown: The cooldown time of the skill in seconds.
        - effect: A description of what the skill does.
        - effects: A list of passive effects. Each effect has:
        - name: The name of the effect.
        - chance: The percentage chance for the effect to trigger.
        - duration: The duration of the effect in seconds.
        - description: A description of the effect.

        The output must be valid JSON. all the descriptions should be within 50 words.
        """
    )

action_router_template = PromptTemplate(
    input_variables=["npc", "env"],
    template="""
        You are an NPC in a simulation-style game similar to AITown or Auto Chess. You should act in a believable, consistent way based on your personality and goals.

        Your profile:
        - Name: {npc[name]}
        - Role: {npc[role]}
        - Personality traits: {npc[traits]}
        - Goals: {npc[goals]}
        - Mood: {npc[mood]}
        - Health: {npc[health]}
        - Inventory: {npc[inventory]}

        Current environment:
        - Location: {env[location]}
        - Nearby characters: {env[nearby_characters]}
        - Notable events: {env[events]}
        - Time of day: {env[time_of_day]}
        - Weather: {env[weather]}

        Based on all the above, answer the following:

        1. What is your current emotion?
        2. What do you say (if anything)?
        3. What action do you take next?
        4. Why did you choose this action?

        Respond in this format:
        {{
        "emotion": "...",
        "dialogue": "...",
        "action": "...",
        "reasoning": "..."
        }}
    """
) 

select_action_template_qwen = PromptTemplate(
    input_variables=["parent_goal", "current_goal", "status", "information","tools", "environment", "recent_actions"], 
    template="""
        You are an assistant, according to the situation that player encountered, choose the appropriate aciton player can take.

        this is the hero status:
        {status}

        this is the environment:
        {environment}

        Here is what you know about the world:
        {information}

        the followng is the actions user can take and thier description:
        {tools}
        which is most appropriate action the for the situation? and you can only use the possible calls of the tool.

        The current parent goal is:{parent_goal}.  
        The current goal is {current_goal}.  
        The recent actions were: {recent_actions}

        Your response should be a JSON object with the following keys, 
        {{
            "tool": "func_name",
            "params":  {{
                    "param1": "value1",
                    "param2": "value2",
                    "param3": "value3",
                }},
            "progress": "true/false",
            "sub_goal": "sub_goal",
            "reason": "reason",
        }}
        
        and think:
        - Can the current goal make pany progress?  
        - If it can make progress now, choose an action, progress session should be true, and sub_goal should be empty.  
        - only if the current goal can't make any progress, provide a new sub-goal.

        Tell me the reason you chose the action. the reason should be concise and clear within 1-2 sentences.
        """

    )

parameter_parse_template_qwen = PromptTemplate(
    input_variables=["hero_name","reason", "tool", "parameters", "goal"], 
    template="""
        this is the tool function {hero_name} takes:
        {tool}
        
        this is the goal:
        {goal}

        this is the {hero_name} situation:
        {reason}

        the following is the function input parameters:
        {parameters}
        
        parse the parameters into json format, and the parameters should be valid json format.
        """
    )

parameter_parse_template = PromptTemplate(
    input_variables=["input", "reason", "tool", "parameters"], 
    template="""
        You are an assistant, according to the player's situation and action, 
        decides the input parameters
        
        this is the player situation:
        {input}
        
        this is the tool function player takes:
        {tool}

        and what player is thinking is:
        {reason}

        the following is the function input parameters:
        {parameters}
        
        and tell me the prediction after the action. the prediction should be concise and clear within 1-2 sentences.
        turn the parameters into json format, and the parameters should be valid json format.
        """
    )
        # this is the actions player can take:
        # {tools}
        # Goal tree:
        # Current goal: {parent_goal}
        # Last goal: {current_goal} {current_goal_problem}
        # Last action result {flag}:
        # {last_action_result}

update_goal_template = PromptTemplate(
    input_variables=["status", "environment", "clue", "current_goal", "recent_actions", "tools"], 
    template="""
        You are a hero in a goal-driven RPG-like world, you are making goal. 

        this is the hero status:
        {status}

        this is the environment:
        {environment}
       
        current task is "defeat the Demon Lord"

        Task Instructions:
        1. You can communicate with the people to gain more information. Multiple conversations won't bring any new information.
        2. If the character already owns or equips the item, do not repeat the same purchase.
        3. Always consider using items in the bag when they may improve the chance of success.
        4. You can improve your power level by equipping equipments or using consumable items(etc. potion, food).
        5. You can only defeat the enemy whose power level is lower than you, or you will die!

        and this is the relevant clue about task:
        {clue}

        Recent actions with their goals and results:
        {recent_actions}
        If there are repetitive actions, pay attention to thinking about whether you are stuck and try to change the goal.

        And resopnd should include:
        1. Give a new sub-goal according to the last_action_result and Last goal and Current goal.
        2. Focus on practical planning and in-world constraints. Consider valid items, locations, money, NPCs, and known world facts.
        3. The item, person, or location mentioned in sub_goal should be valid in the world, only use terms in World Knowledge.
        4. And tell me the reason about your response. the reason should be concise and clear within 1-2 sentences.
        Your response should be a JSON object with the following keys, 
        {{
            "new_sub_goal": "a new sub_goal",
            "reason": "reason",
        }}
        """
    )

        # this is the hero status:
        # {status}

        # this is the environment:
        # {environment}

select_action_template_qwen = PromptTemplate(
    input_variables=["current_goal", "status", "reason", "information","tools", "environment", "recent_actions"], 
    template="""
        You are a hero in a goal-driven RPG-like world, choose the appropriate aciton player can take.

        hero status:
        {status}

        Current goal is {current_goal}
        reason: {reason}

        Here is what you know about the world:
        {information}

        the followng is the possible actions hero can take and thier description:
        {tools}
        which is most appropriate action the for the situation? and you can only use the possible calls of the actons.

        Your response should be a JSON object with the following keys, 
        {{
            "action": "acton_name",
            "params":  {{
                    "param1": "value1",
                    "param2": "value2",
                    "param3": "value3",
                }},
            "reason": "reason",
        }}

        Tell me the reason you chose the action. the reason should be concise and clear within 1-2 sentences.
        """
    )