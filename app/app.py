import os
import streamlit as st

from dotenv import load_dotenv
from settings import AI_LLM_MODELS
from utils.general import setup_env, load_openai_model
from utils.ui import intro, single_article, multiple_articles

# Page title    
st.set_page_config(page_title="🦜🔗 Thesis AI Assistat")
st.title("🦜🔗 Thesis AI Assistant")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.text_input('Open AI API key')
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    setup_env()
    model_name = st.sidebar.selectbox(
        "Which LLM model would you like to use?",
        AI_LLM_MODELS,
        index=5,
        placeholder="Select model...",
        key="sidebar_model",
    )
    st.sidebar.write("You selected:", model_name)
    model = load_openai_model(model_name)

    page_names_to_funcs = {
        "Intro": intro,
        "Single Article": single_article,
        "Multiple Articles": multiple_articles,
    }
    fun_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())

    if fun_name != "Intro":
        page_names_to_funcs[fun_name](model)
    else:
        page_names_to_funcs[fun_name]()