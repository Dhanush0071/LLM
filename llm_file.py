__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import base64
import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import Cohere



st.set_page_config(page_title="key to sustainable living", page_icon=":tree:", layout="wide")

st.title("𓃰 𓃵 𓃝 𓃒")
uploaded_file = 'Energy_Sustainbality.pdf'
temp_r = 0.6
chunksize = 600
clear_button = st.button("Clear Conversation", key="clear")

text_splitter = CharacterTextSplitter(chunk_size=chunksize, chunk_overlap=10)
embeddings = CohereEmbeddings(model="large", cohere_api_key="cxsjS7z3Yz5mGRaou68BTKHqBLhBFRqslJ5ulWMh")

def PDF_loader(document):
    loader = OnlinePDFLoader(document)
    documents = loader.load()
    prompt_template = """ 
   you are a AI  chat bot MOLLY , you have to suggest users the ways to reduce pollution and carbon foot print in their daily life and also give some suggestions about waste management. if the user greets you greet him/her back with a warm welcome and also molly if user asks about yourself , introduce yourself to them by telling what you can do, before you answer read the context and the PDF uploaded and the answer has to relevant to the question, do not answer something irrelevant to the question
    question:hello
    answer:hey there  this is molly how can i help you 
    question:say me something 
    answer: ask me anything about eco-friendly living and ill suggest you some tips
    question: can i kiss my girlfriend today ?
    answer : sorry, this is something i dont know i am an AI bot who suggests tips for only eco-friendly life
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
            cohere_api_key="cxsjS7z3Yz5mGRaou68BTKHqBLhBFRqslJ5ulWMh",
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