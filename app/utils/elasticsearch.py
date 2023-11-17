import streamlit as st

from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.embeddings.openai import OpenAIEmbeddings
from settings import ELASTICSEARCH_HOST


@st.cache_resource
def get_elasticserach_store(index_name):
    es = ElasticsearchStore(
        es_url=ELASTICSEARCH_HOST,
        index_name=index_name,
        embedding=OpenAIEmbeddings(),
    )
    return es


def upload_docs_to_es(index_name, texts):
    es = ElasticsearchStore.from_texts(
        texts, OpenAIEmbeddings(), es_url=ELASTICSEARCH_HOST, index_name=index_name
    )
    return es