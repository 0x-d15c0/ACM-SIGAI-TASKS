# RAG Chatbot Project
This project implements a Retrieval-Augmented Generation (RAG) chatbot using the Ollama model with FAISS for document retrieval. The chatbot is designed to answer user queries based on the content of a set of PDF documents. A Streamlit interface is used to provide an interactive chat experience.

* Ollama Phi3 Model: The language model used for generating responses.
* HuggingFace Bge Embeddings: Used for embedding the document chunks.
* PyPDFDirectoryLoader: Loads PDF documents from the entered directory.
* RecursiveCharacterTextSplitter: Splits documents into smaller chunks.
* FAISS: Vector store used for efficient similarity search.
* PromptTemplate: Defines the format for the prompt.
* RetrievalQA: Combines document retrieval and question answering to generate responses.
* Frontend displays chat history
  
# Installation
#### 1. Clone the repository
```sh
git clone https://github.com/your-repo/rag-chatbot.git
cd rag-chatbot
```
#### 2. Install the required packages
```sh
pip install -r requirements.txt
```
#### 3. Usage - Run the Streamlit application:
```sh
streamlit run frontend.py
```
![image](https://github.com/0x-d15c0/ACM-SIGAI-TASKS/assets/117750351/afe91928-a65c-490d-8841-abe9c012a4ec)
