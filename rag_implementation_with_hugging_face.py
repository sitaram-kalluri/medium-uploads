# Install the required python packages.
!python -m pip install --upgrade pip
!pip install -U chromadb langchain-community transformers tiktoken sentence-transformers torch

import os
import torch
from operator import itemgetter
from langchain_community.document_loaders import PyPDFLoader;
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

def load_data_to_db():
    loader = PyPDFLoader("sample_document.pdf")
    pdf_data = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=50)
    split_data = text_splitter.split_documents(pdf_data)

    persist_directory = os.path.join(os.getcwd(), local_directory)

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vect_db = Chroma.from_documents(split_data,
                                    embedding_function,
                                    collection_name="sample_collection",
                                    persist_directory="sample_embedding"
                                    )
    vect_db.persist()
    return vect_db

def get_llm():
    # Define the model name and retrieve the necessary token for authentication.
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    #For security purposes, store the token key in an environment variable and access it from there.
    token = os.getenv("API_TOKEN_KEY")

    # Load the model and tokenizer from Hugging Face with the specified configurations.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token=token,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    # Create a pipeline for text generation using the loaded model and tokenizer.
    llama_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=llama_pipeline, model_kwargs={'temperature': 0.7})

    return llm

# Function to format a list of documents into a single string.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to ask a question and receive an answer using the large language model and the document database.
def ask_model(vector_db, llm, question):
    # Define a template for the prompt to be used with the large language model.
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer as concise as possible and indent the code example if any into the new lines.
    {context}
    Question: {question}
    Helpful Answer:"""
    custom_prompt = PromptTemplate.from_template(template)

    # Create a chain of operations to process the question.
    rag_chain_from_docs = (
            {
                "context": lambda input: format_docs(input["documents"]),
                "question": itemgetter("question"),
            }
            | custom_prompt
            | llm
    )
    rag_chain_with_source = RunnableMap(
        {
            "documents": vector_db.as_retriever(),
            "question": RunnablePassthrough()}
    ) | {
                                "documents": lambda input: [doc.metadata for doc in input["documents"]],
                                "answer": rag_chain_from_docs,
                            }

    # Invoke the chain of operations with the question.
    response = rag_chain_with_source.invoke(question)
    return response

if __name__ == "__main__":
    db = load_data_to_db()
    llm = get_llm()
    prompt = input("Enter your question: ")
    while prompt.lower() != 'bye':
        # Adding new line character for line seperation between question and answer
        print('\n')
        response = ask_model(db,llm, prompt)
        helpful_answer = response['answer'].split('Helpful Answer:')[1].strip()
        # Printing the "Helpful Answer"
        print(helpful_answer)
        # Adding new line character for line seperation between question and answer
        # Take the next user input
        prompt = input("\nEnter your question: ")

    print("Visit us again")
