import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
#from langchain.vectorstores import FAISS   llx-BpE4gU37E6w7A9pQ2p8jeJxiQDME0EQZT95PvygITWvwYd9J
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
#from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from llama_parse import LlamaParse

load_dotenv()

parser = LlamaParse(
    api_key="llx-BpE4gU37E6w7A9pQ2p8jeJxiQDME0EQZT95PvygITWvwYd9J",  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="text"  # "markdown" and "text" are available
)
# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="sales_response.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
#llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
llm = Ollama(model="llama3.1:8b",)
template = """
 {message}
"""
msg = ''
# You are a world class business development representative. 
# I will share a prospect's message with you and you will give me the best answer that 
# I should send to this prospect based on past best practies, 
# and you will follow ALL of the rules below:

# 1/ Response should be very similar or even identical to the past best practies, 
# in terms of length, ton of voice, logical arguments and other details

# 2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

# Below is a message I received from the prospect:
# {message}

# Here is a list of best practies of how we normally respond to prospect in similar scenarios:
# {best_practice}

# Please write the best response that I should send to this prospect:


prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response



# 5. Build an app with streamlit
def main():
    global msg
    st.set_page_config(
        page_title="Customer response generator", page_icon=":bird:")

    st.header("Customer response generator :bird:")
    message = st.text_area("customer message")

    if message:
        st.write("Generating best practice message...")
        msg = msg + "\n user:" + message
        message = "This is the earlier conversation:" + msg + "\n try answer to the user"
        result = generate_response(message)
        msg = msg + "\n LLM:" + result
        st.info(result)


