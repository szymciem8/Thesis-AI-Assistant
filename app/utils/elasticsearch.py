import streamlit as st

from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.embeddings.openai import OpenAIEmbeddings
from settings import ELASTICSEARCH_HOST

embeddings = OpenAIEmbeddings()

@st.cache_resource
def get_elasticserach_store(index_name):
    es = ElasticsearchStore(
        es_url=ELASTICSEARCH_HOST,
        index_name=index_name,
        embedding=embeddings,
    )
    return es


def upload_docs_to_es(index_name, texts):
    es = ElasticsearchStore.from_texts(
        texts, embeddings, es_url=ELASTICSEARCH_HOST, index_name=index_name
    )
    return es