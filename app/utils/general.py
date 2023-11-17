import os
import sys
import openai
import streamlit as st

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import AI_CHAT_MODELS


@st.cache_resource
def load_openai_model(model_name):
    if model_name in AI_CHAT_MODELS:
        return ChatOpenAI(model=model_name)
    return OpenAI(model=model_name)

def setup_env():
    openai.api_key = os.environ["OPENAI_API_KEY"]

def check_openai_api_key(api_key):
    openai.api_key = api_key
    try:
        model = openai.OpenAI()
    except:
        return False
    return True