# -*- coding: utf-8 -*-

import os
import wget
import streamlit as st
import streamlit_themes as stt
from streamlit_chat import message
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import Cohere

# wget.download("https://github.com/Dhanush0071/LLM/blob/main/Energy%20Sustainbality.pdf",'Energy_Sustainbality.pdf')


st.set_page_config(page_title="key to sustainable living", page_icon=":tree:")
stt.set_theme('sky')

st.markdown(
    "<h1 style='text-align: center;'>LETS LEAD AN ECO-FRIENDLY LIFE</h1>",
    unsafe_allow_html=True,
)

with st.sidebar:
    uploaded_file = 'Energy_Sustainbality.pdf'
    temp_r = st.slider("Temperature", 0.1, 0.9, 0.3, 0.1)
    chunksize = st.slider("Chunk Size for Splitting Document ", 256, 1024, 300, 10)
    clear_button = st.button("Clear Conversation", key="clear")

text_splitter = CharacterTextSplitter(chunk_size=chunksize, chunk_overlap=10)
embeddings = CohereEmbeddings(model="large", cohere_api_key="vLuTQVcIyLBLbb5UqNJb4sFitqv1D2g8mriKoFoi")

def PDF_loader(document):
    loader = OnlinePDFLoader(document)
    documents = loader.load()
    prompt_template = """ 
    You are MOLLY a AI Chatbot developed to provide users with tips and suggestions for leading a sustainable and an eco friendly life by telling them the tips to reduce global warming and reducing their carbon footprint by the context provided. Based on the information in the attached PDF, you can offer tailored recommendations for sustainable living practices that can help users to reduce their impact on the environment. Use the following pieces of context to answer the question at the end.do a friendly conversation if they greets you. Greet Users!!
    S:hello
    E:Hey there this is MOLLY , How can i help you 
    {context}
    {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    texts = text_splitter.split_documents(documents)
    global db
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    global qa
    qa = RetrievalQA.from_chain_type(
        llm=Cohere(
            model="command-xlarge-nightly",
            temperature=temp_r,
            cohere_api_key="vLuTQVcIyLBLbb5UqNJb4sFitqv1D2g8mriKoFoi",
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    return "Ready"

PDF_loader("Energy_Sustainbality.pdf")
st.markdown(
"<h3 style='text-align: center;'>Hello i am MOLLY an Eco-Friendly instructor bot "
+ "</h3>",
unsafe_allow_html=True,)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

def generate_response(query):
    result = qa({"query": query, "chat_history": st.session_state["chat_history"]})
    return result["result"]

response_container = st.container()
container = st.container()

with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_input("You:", key="input")
        submit_button = st.form_submit_button(label="Send")

    if user_input and submit_button:
      output = generate_response(user_input)
      print(output)
      st.session_state["past"].append(user_input)
      st.session_state["generated"].append(output)
      st.session_state["chat_history"] = [(user_input, output)]

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(
                st.session_state["past"][i],
                is_user=True,
                key=str(i) + "_user",
                avatar_style="adventurer",
                seed=123,
            )
            message(st.session_state["generated"][i], key=str(i))

if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["chat_history"] = []
