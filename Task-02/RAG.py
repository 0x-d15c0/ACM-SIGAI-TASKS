from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def rag(question):

    model = Ollama(model="phi3")

    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    loader = PyPDFDirectoryLoader("./pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)

    prompt_template = """
    You are an intelligent assistant with access to a large set of documents. Answer the following question based on the information from these documents.

    Context: {context}

    Question: {question}

    Answer:
    """

    vectorstore = FAISS.from_documents(final_documents[:200], huggingface_embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retrievalQA = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = retrievalQA.invoke({"query": question})
    return result['result']

