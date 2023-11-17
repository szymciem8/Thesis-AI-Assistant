import streamlit as st
import requests
import base64
import sys

from pathlib import Path

from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.elasticsearch import ElasticsearchStore

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.custom_parsers import SemicolonSeparatedListOutputParser
from utils.elasticsearch import get_elasticserach_store
from settings import SINGLE_ARTICLE_INDEX, ARTICLES_INDEX


class ScientificPDFBase:
    def __init__(self, model):
        self.model = model
        self._knowledge_base = None

    @property
    def _retriever(self):
        raise NotImplementedError

    @st.cache_resource
    def _load_pdf(_self, url):
        return PyPDFLoader(url)
    
    def _get_or_create_knowledge_base(self, index_name=None):
        if not self._knowledge_base:
            try:
                self._knowledge_base = get_elasticserach_store(index_name=index_name)
            except:
                self._knowledge_base = Chroma(collection_name=index_name, embedding_function=OpenAIEmbeddings())
        return self._knowledge_base

    def vector_db_type(self):
        if type(self._knowledge_base) is Chroma:
            return 'Chroma'
        if type(self._knowledge_base) is ElasticsearchStore:
            return 'Elsticsearch'
        return None

    def ask(self, query):
        """
        Ask question about the document based on the retriever context.
        """
        qa = RetrievalQA.from_chain_type(
            llm=self.model, chain_type="stuff", retriever=self._retriever
        )
        return qa.run(query)


class ScientificPDF(ScientificPDFBase):
    def __init__(self, url, model):
        super().__init__(model)
        self.url_path = url
        self.pdf = self._load_pdf(url)
        self.pages = self.pdf.load_and_split()
        self.text = self._extract_text()
        self._knowledge_base = self._get_or_create_knowledge_base(index_name=SINGLE_ARTICLE_INDEX)
        self._create_knowledge_base()

    @property
    def _retriever(self):
        return self._knowledge_base.as_retriever()

    def _create_knowledge_base(self):
        """
        Create ChromaDB knowledge base
        """
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(self.text)
        self._knowledge_base.add_texts(chunks)

    def _extract_text(self):
        """
        Extract text from PDF pages.
        """
        text = ""
        for page in self.pages:
            text += page.page_content
        return text

    def display_pdf(self):
        """
        Display PDF in streamlit embedded in markdown script.
        """

        pdf_base_64 = self.pdf_url_to_base64()
        pdf_display = f'<embed src="data:application/pdf;base64,{pdf_base_64}" width="700" height="1000" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)

    def pdf_url_to_base64(self):
        """
        Cast PDF file to base64
        """
        try:
            response = requests.get(self.url_path, stream=True)
            pdf_base64 = base64.b64encode(response.content).decode("utf-8")
            return pdf_base64

        except requests.exceptions.RequestException as e:
            print(f"Error fetching PDF from URL: {e}")
            return None

    def summarize(self):
        """
        Summarize text from PDF using an LLM.
        """
        summary_chain = load_summarize_chain(llm=self.model, chain_type="map_reduce")
        summarize_document_chain = AnalyzeDocumentChain(
            combine_docs_chain=summary_chain
        )
        return summarize_document_chain.run(self.text)

    def generate_keywords(self, n_keywords):
        """
        Generate choosen number of keywords based on the article.
        """
        template = """
        Based on the following pieces of text: \n {text} \n
        create {n_keywords} keywords that represent the text in the best way possible.
        Return it in a form of a list. \n
        For example: keyword1, keyword2, keyword3
        """
        output_parser = CommaSeparatedListOutputParser()
        prompt = PromptTemplate(
            template=template,
            input_variables=["text", "n_keywords"],
            output_parser=output_parser,
        )
        chain = LLMChain(llm=self.model, prompt=prompt)
        output = chain.run(text=self.text, n_keywords=n_keywords)
        return output_parser.parse(output)

    def _get_highlights(self, query):
        """
        Get exact parts of the text based on the query
        """
        template = """
        Generate a list of texts which are described by this query: {query}.
        The list can consist of couple of elements or event one if it's sufficient.
        The list consists of EXACT part of the text that can be found it that text. 
        It has to be perfectly precise so it could be highlighted. 
        By precise I mean exact part of the text so it can be found. 
        No character can be different from the original text. \n

        This is the text: {text}
        """
        output_parser = SemicolonSeparatedListOutputParser()
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "text"],
            output_parser=output_parser,
        )
        chain = LLMChain(llm=self.model, prompt=prompt)
        output = chain.run(query=query, text=self.text)

        list_of_highlights = output_parser.parse(output)
        exact_texts = []
        for highlight in list_of_highlights:
            retrieved_docs = self._retriever.invoke(highlight)
            sub_text = (
                retrieved_docs[0].page_content.replace("-\n", "").replace("\n", " ")
            )
            exact_texts.append(sub_text)
        return exact_texts

    def show_highlighted_sections(self, query):
        sections = self._get_highlights(query)
        return sections


class MutipleScientificPDFs(ScientificPDFBase):
    def __init__(self, model):
        super().__init__(model)
        self._knowledge_base = self._get_or_create_knowledge_base(ARTICLES_INDEX)

    @property
    def _retriever(self):
        """
        Get retriever from Elasticsearch retriever
        """
        retriever = self._get_or_create_knowledge_base(ARTICLES_INDEX).as_retriever()
        return retriever

    def _split_list_str(self, string):
        return string.split(",")

    def _get_pdfs_text(self):
        text = ""
        for url in self.url_list:
            pdf = PyPDFLoader(url)
            pages = pdf.load_and_split()
            for page in pages:
                text += page.page_content
        return text

    def upload_pdfs(self, url_list):
        """
        Upload pdfs to Elasticsearch
        """
        self.url_list = self._split_list_str(url_list)
        self._update_vec_db()

    def _update_vec_db(self):
        """
        Create ElasticSearch knowledge base
        """
        text = self._get_pdfs_text()
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=500, chunk_overlap=100, length_function=len
        )
        chunks = text_splitter.split_text(text)
        self._knowledge_base.add_texts(chunks)