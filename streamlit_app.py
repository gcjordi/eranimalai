import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

openai.api_key = st.secrets.openai_key

st.set_page_config(page_title="ErAnimal - AIVet", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Tu servicio veterinario asistido por Inteligencia Artificial")
st.info("Este es el agente experto de IA sin coste que atiende consultas veterinarias")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Realiza consultas sobre salud, cuidado y bienestar animal!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the knowledge bases docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-16k", temperature=0.1, system_prompt="You are an AI agent who handles veterinary consultations like a highly skilled veterinary professional does. At the same time, you will be able to provide answers to general questions related to the health, care and well-being of animals."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, system_prompt="You are an AI agent who handles veterinary consultations like a highly skilled veterinary professional does. At the same time, you will be able to provide answers to general questions related to the health, care and well-being of animals.")
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
