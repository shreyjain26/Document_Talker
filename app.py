import streamlit as st
from utils import get_content, rag_chain, chat
import time


st.title("Document Talker")


if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = set()
if 'data' not in st.session_state:
    st.session_state.data = []


with st.sidebar:
    st.sidebar.header("Upload PDF Documents")

    uploaded_files = st.sidebar.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True, key="file_uploader")
    weblinks = st.sidebar.text_area("Enter Web Links (comma-separated)", key="weblinks")
    mode = st.sidebar.selectbox("Select Mode", ["Doc", "Web", "combined"])

    if st.button("Submit and Process"):
        with st.spinner('Extracting content...'):
            data = get_content(uploaded_files, weblinks=weblinks, mode=mode)
            st.session_state.data.append(data)
            st.success("Done")

        st.toast(f"Extracted content", icon="📄")
        time.sleep(4)


if 'data' in st.session_state and st.session_state.data:

    llm_chain = rag_chain(st.session_state.data)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        response_placeholder = st.empty()
        full_response = ""

        with st.spinner('Generating response...'):
            stream = chat(prompt, llm_chain)

            for chunk in stream['answer'].split(' '):
                full_response += chunk + " "
                time.sleep(0.10)
                response_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


else:
    st.info("Upload PDF documents to get started.")