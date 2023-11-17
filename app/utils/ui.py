import streamlit as st

from utils.pdf import ScientificPDF, MutipleScientificPDFs
from validators import url

def intro():
    import streamlit as st

    st.sidebar.success("Select a page for specific functionality.")

    st.markdown(
        """
        Thesis AI Asistant is an open-source app createad for analysis of scientific research
        to help to write more papers. 

        **👈 Select from the dropdown on the left** to see what AI Assitant can do!

        ### Want to learn more about Streamlit?

        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)
    """
    )


def single_article(model):
    url_str = st.text_input("Article URL to PDF")
    pdf = None
    if url(url_str) and model:
        pdf = ScientificPDF(url_str, model)

    chat_tab, basics_tab, pdf_viewer_tab, highlights_tab, listen_tab = st.tabs(["📈 Chat", "📝 Basics", "🔎 PDF Viewer", "💡 Highlights", "🗣️ Listen"])

    with chat_tab:
        show_single_pdf_chat(pdf)

    with basics_tab:
        show_basics(pdf)

    with pdf_viewer_tab:
        show_pdf_viewer(pdf)

    with highlights_tab:
        show_highlights(pdf)

    with listen_tab:
        show_listen(pdf)
        # st.info("Coming soon!")


def multiple_articles(model):
    with st.form(key="Upload documents"):
        url_list = st.text_input("Paste list of URLs to scientific articles (urls must be comma separated)")
        multiple_pdfs = MutipleScientificPDFs(model)
        db_type = multiple_pdfs.vector_db_type()
        if db_type == "Chroma":
            st.info("Elasticsearch is not available. You're using Chroma.")
        submit_docs = st.form_submit_button("Submit documents")
        if submit_docs:
            multiple_pdfs.upload_pdfs(url_list)
    show_multiple_pdfs_chat(multiple_pdfs)


def show_multiple_pdfs_chat(multiple_pdfs):
    if "multiple_pdf_messages" not in st.session_state.keys():
        st.session_state.multiple_pdf_messages = [{"role": "assistant", "content": "I'm your Thesis AI Assistant, How may I help you?"}]
    prompt = st.text_input("Ask a question about your articles:", disabled=not multiple_pdfs)
    if prompt:
        st.session_state.multiple_pdf_messages.append({"role": "user", "content": prompt})
    if st.session_state.multiple_pdf_messages[-1]["role"] != "assistant":
        response = multiple_pdfs.ask(prompt)
        message = {"role": "assistant", "content": response}
        st.session_state.multiple_pdf_messages.append(message)
    for message in st.session_state.multiple_pdf_messages[::-1]:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def show_single_pdf_chat(pdf):
    if "single_pdf_messages" not in st.session_state.keys():
        st.session_state.single_pdf_messages = [{"role": "assistant", "content": "I'm Thesis AI Assistant, How may I help you?"}]
    prompt = st.text_input("Ask a question about your article:", disabled=not pdf)
    if prompt:
        st.session_state.single_pdf_messages.append({"role": "user", "content": prompt})
    if st.session_state.single_pdf_messages[-1]["role"] != "assistant":
        response = pdf.ask(prompt)
        message = {"role": "assistant", "content": response}
        st.session_state.single_pdf_messages.append(message)
    for message in st.session_state.single_pdf_messages[::-1]:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def show_basics(pdf):
    with st.form(key="Summary"):
        st.header("Summary", divider="rainbow")
        submitted = st.form_submit_button("Generate summary", disabled=not pdf)
        if submitted:
            st.info(pdf.summarize())
        else:
            st.text("Place for article summary")
    with st.form(key="Keywords"):
        st.header("Key words", divider="rainbow")
        n_of_keywords = st.slider(
            "How many keywords would you like to generate?", 0, 25, 5
        )
        submitted = st.form_submit_button("Generate keywords", disabled=not pdf)
        if submitted:
            keywords = pdf.generate_keywords(n_of_keywords)
            st.info(", ".join(keywords))
        else:
            st.text("No generated keywords")


def show_pdf_viewer(pdf):
    if pdf:
        pdf.display_pdf()
    else:
        st.text("Upload URL to PDF to view it here")


def show_highlights(pdf):
    with st.form(key="Highlights"):
        query = st.text_input("Write which part of the text you want to highlight", disabled=not pdf)
        submitted = st.form_submit_button("Generate highlights")
        if submitted and query:
            st.info(pdf.show_highlighted_sections(query))


def show_listen(pdf):
    generated = False
    with st.form(key="Listen"):
        n_of_words = st.slider(
            "How long should your speech be? This is a maximum possible value.", 200, 1000, 250, step=10
        )
        lector_name = st.selectbox(
            "Choose your lector",
            ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            index=0,
            placeholder="Select voice...",
            key="sidebar_lector",
        )
        submitted = st.form_submit_button("Generate speech")
        if submitted:
            generated, text = pdf.text_2_speech(n_of_words, voice=lector_name)
            st.info(text)
            if generated:
                audio_file = open('output.mp3', 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')