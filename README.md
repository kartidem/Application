#  IDEMIAGPT

## Introduction
This project is a **User Manual Q&A Interface** for Parameters and Admin Guide. It is designed to extract and process the text from various file formats like PDF, DOCX and TEXT. It utilizes LangChain and Google GenAI for natural language processing, enabling users to ask questions about the content of these documents and recieve detailed answers.

The main interface allows users to select either the **Admin Guide** or the **Parameter Guide**(present in sidebar) and ask questions from it. When a guide is selected, the application processes it and generates text embeddings and stores the embeddings in a vector store. The vector store is then used to facilitate similarity searches for user queries. User questions are logged and responses are generated.

This project is ideal for creating an interactive and intelligent help desk for user manuals, making it easier to navigate complex documentation and find specific information quickly.


## Workflow Diagram

![image](https://github.com/user-attachments/assets/d43a145c-24a2-47d9-b49b-d1ad7b0e0994)



## Source Code
This section consists of a concise explaination of the entire source code.

- `get_text_from_file(file_path)`
  - This function extracts text content from the chosen Guide
  - Depending on the file type of the guide (pdf by default) it uses appropriate libraries to read the file and extract text. For **PDF** files it uses `PyPDF2.PdfReader`
  - If text extraction fails, it raises a `TextExtractionException`
- `get_text_chunks(raw_texts)`
  - This function splits large texts into smaller manageable chunks to improve model efficiency.
  - It uses `RecursiveCharacterTextSplitter` from **LangChain** to split the raw text into chunks of size 10,000 characters with an overlap of 1,000 characters
  - Returns a list of text chunks
- `get_vectorstore(text_chunks)`
  - Creates and stores text embeddings in a **FAISS** vector store.
  - Uses `GoogleGenerativeAIEmbeddings` to generate embeddings for each text chunk. Embedding model: `embedding-001`
  - The embeddings created are vectors containing float values which represents the position of the text chunk in a high-dimensional space. Each dimension in the embedding vector captures a different aspect of the chunk's semantics.
  - Stores these embeddings in a **FAISS** vector store
  - Saves the vector store locally
  - Returns the vector store
- `get_convo_chain(vectorstore)`
  - Sets up a conversational AI chain using the vector store
  - Defines a prompt template to structure the input for generative AI model
  - Initializes the `ChatGoogleGenerativeAI` model : `gemini-1.5-pro`
  - Loads a question-answering chain using LangChain `load_qa_chain` function with the specified model and prompt
  - Returns the conversational chain
- `handle_userinput(user_question)`
  - Processes user questions and generates responses
  - Loads the FAISS vector store and performs a similarity search to find relevant text chunks
  - Uses conversational AI chain to generate a response based on the retrieved text chunks
  - Appends the user question and generates response to the chat history
  - Displays chat history in the streamlit interface using custom HTML and CSS templates to style it for user friendliness.


## Dependencies 
Below mentioned are the dependencies:
- `streamlit==1.11.0`
- `python-dotenv==1.0.1`
- `PyPDF2==3.0.1`
- `langchain==0.2.1`
- `faiss-cpu==1.8.0`
- `langchain-google-genai==1.0.5`
- `langchain-text-splitters==0.2.0`
- `google-generativeai==0.5.4`

## Files and folders
- faiss_index: contains the vector store in pkl file format
- Administrator_Guide.pdf: Admin Guide pdf from where text is extracted
- Bio_types.txt: Contains all the thrift commands related to biotypes
- Common_Generic_Commands.txt: Contains all the common thrift commands used
- IDEMIALogoTransparent.jpg: Used as the logo
- MA5G_Generic_commands: Contains all the MA5G commands
- Parameter_Guide.pdf: Parameter Guide pdf from where text is extracted
- appl.py: Source code
- botPic.png: Bot pfp in the messages
- htmlTemplates.py: Contains the html and css scripts for styling the messages
- log.txt: log of username and the questions they ask
- requirements.txt: List of dependencies
- security_types.txt: Contains all the security thrift commands
- userPic.png: user pfp in the messages

## Links
- Streamlit cloud link : https://share.streamlit.io/
- App Link : https://idemiagpt.streamlit.app/

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Quick Setup:

1. Checkout the code from github public repo "Application" (credentials required)

2. Change the code in appl.py on local and checkin to git hub.

3. Invoke the application hosted on streamlit web by accessing AppLink, the code will be fetched from git and executed.
     https://idemiagpt.streamlit.app/



/////////////////////////////////////////////////////////////////////////////////////////
Deploy and execute the application on local machine:

1. Install the python >=3.9.2  on your local machine.
   Check python version executing python --version. if version is not correct add the path of python.exe in environment varaible.
2. Update C:\Users\<GID>\AppData\Roaming\pip\pip.ini with below mentioned text if you are in idemia network.
    
	   [global]
       timeout = 600
       index = https://nexus.os.plf.idemia.com/repository/pypi/pypi
       index-url = https://nexus.os.plf.idemia.com/repository/pypi/simple
       trusted-host = nexus.os.plf.idemia.com
	   
3. Install virtualenv using command pip install virtualenv
4. Create virtual environment :
    >python -m virtualenv env_gpt
	>cd <path to env_gpt>
	>cd Scripts
	>activate
	(env_gpt)>cd <path to Application folder>
	(env_gpt)>pip install -r requirements.txt
	
	All the dependencies will be installed in the virtual environment.
	
5. Not recommended but you have to add GOOGLE_API_KEY in windows environment variable to make app work on local machine.

6. API_KEY is stored in streamlit secret store and it can be found under settings option on https://share.streamlit.io/
 
7. comment #api_key = st.secrets["GOOGLE_API_KEY"]
   instead add  api_key = os.environ["GOOGLE_API_KEY"]
   
8. Change the code in appl.py on local and execute the below command to access the app from local
   
    (env_gpt)>streamlit run appl.py
	 
	  currently facing google.api_core.exceptions.DeadlineExceeded: 504 Deadline Exceeded error 
	  perhaps due to access blocked from idemia network.
	  
	  Try to use public internet.
	
