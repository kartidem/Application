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

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

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
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)
    docs = vectorstore.similarity_search(user_question)
    response = st.session_state.conversation({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    output_text = response.get('output_text', '')

    if not output_text.strip():
        output_text = "Encountered an error while generating response. Please enter the prompt again! Try re-arranging the prompt. For example if the prompt was \"parameter to configure MMI LED status\", try again with \"configure MMI LED status\""
    

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "bot", "content": output_text})

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
    if "selected_button" not in st.session_state:
        st.session_state.selected_button = None

    
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
        logo_html = """
        <style>
            #idemia-logo{
                margin-bottom: 20%;
                width: 300px;
                height: 75px;
            }
        </style>
        <img src="data:image/png;base64,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" id="idemia-logo">
        
        """
        st.markdown(logo_html, unsafe_allow_html=True)

        st.subheader("Select the field to ask from:")

    

        if st.button("Admin Guide"):
                st.session_state.chat_history = []
            
                
                try:
                    uploaded_files = ["Administrator_Guide.pdf"] 
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
                uploaded_files = ["Parameters_Guide.pdf"]
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
                uploaded_files = ["Bio_types.txt", 
                                  "Common_Generic_Commands.txt",
                                  "MA5G_Generic_commands.txt", 
                                  "security_types.txt"] 
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
