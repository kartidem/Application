import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import docx
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template

class TextExtractionException(Exception):
    pass
class FileNotFoundException(Exception):
    pass

class ProcessingException(Exception):
    pass

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text_from_file(file_path):
    text = ""
    try:
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                text = json.dumps(data, indent=2)
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        return text
    except TextExtractionException:
        st.write("TextExtractionException: Couldnt extract text from selected files")

def get_text_chunks(raw_texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(raw_texts)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def get_convo_chain(vectorstore):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.3)
    # need active internet connection for the model to work
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
    

def handle_userinput(user_question):
    
    with open(".\log.txt", "a") as log_file:
        log_file.write(f"User Question: {user_question}\n")


    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("faiss_index", embeddings)
    docs = vectorstore.similarity_search(user_question)
    response = st.session_state.conversation({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "bot", "content": response['output_text']})

    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)


def main():

    st.set_page_config(page_title="User Manual Q/A Interface", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    
    st.markdown("""
                
    <style>
        #Log-out{
            position: fixed;
            top: 50px;
            right: 30px;
        }
    </style>
    
    <a id="Log-out" href="http://localhost:3000" target="_self" onclick="window.close()">Logout</a>
    <script>
        document.querySelector('a').onclick = function() {
            setTimeout(function() { window.close(); }, 1000); 
        }
    </script>
    """, unsafe_allow_html=True)

    st.header("User Manual Q/A Interface")
    user_question = st.text_input("Ask your question")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Select the field to ask from:")

        if st.button("Admin Guide"):
                
                st.session_state.chat_history = []
                try:
                    uploaded_files = [".\Dataset\Administrator_Guide.pdf"] 
                except FileNotFoundException(Exception):
                    st.write("FileNotFoundException: File(s) of the following field either dont exist or are corrupted")
                
                try:
                    with st.spinner("Processing..."): 
                        raw_texts=""
                        for uploaded_file in uploaded_files:
                            raw_texts += get_text_from_file(uploaded_file)
                        text_chunks = get_text_chunks(raw_texts)
                        vectorstore=get_vectorstore(text_chunks)
                        #st.session_state.clear()
                        st.session_state.conversation = get_convo_chain(vectorstore)
                except ProcessingException:
                    st.write("ProcessingException: an error was encountered while setting up the model")
                    #st.session_state.chat_history = []

        if st.button("Parameters Guide"):
            st.session_state.chat_history = []
            try:
                uploaded_files = [".\Dataset\Parameters_Guide.pdf"]
            except FileNotFoundException(Exception):
                    st.write("FileNotFoundException: File(s) of the following field either dont exist or are corrupted") 
            try:  
                with st.spinner("Processing..."): 
                    raw_texts=""
                    for uploaded_file in uploaded_files:
                        raw_texts += get_text_from_file(uploaded_file)
                    text_chunks = get_text_chunks(raw_texts)
                    vectorstore=get_vectorstore(text_chunks)
                    #st.session_state.clear()
                    st.session_state.conversation = get_convo_chain(vectorstore)
            except ProcessingException:
                st.write("ProcessingException: an error was encountered while setting up the model")

        if st.button("Thrift Commands"):
            st.session_state.chat_history = []

            try:
                uploaded_files = [".\Dataset\Bio_types.txt", 
                                  ".\Dataset\Common_Generic_Commands.txt",
                                  ".\Dataset\MA5G_Generic_commands.txt", 
                                  ".\Dataset\security_types.txt"] 
            except FileNotFoundException(Exception):
                    st.write("FileNotFoundException: File(s) of the following field either dont exist or are corrupted")

            """Thrift Commands currently only contain
                    - Bio types
                    - Common Generic commands
                    - MA5G Generic Commands
                    - Security Types
            """
            try:
                with st.spinner("Processing..."): 
                    raw_texts="" 
                
                    for uploaded_file in uploaded_files:
                        raw_texts += get_text_from_file(uploaded_file)
                    text_chunks = get_text_chunks(raw_texts)
                    vectorstore = get_vectorstore(text_chunks)
                    #st.session_state.clear()
                    st.session_state.conversation = get_convo_chain(vectorstore)
            except ProcessingException:
                st.write("ProcessingException: an error was encountered while setting up the model")

if __name__ == '__main__':
    main()
