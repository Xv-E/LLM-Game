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
