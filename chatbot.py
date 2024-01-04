from typing import Any
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch
from torch import cuda, bfloat16

custom_prompt_template = """Use the following information to answer the user's question.
In case you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

def load_llama(model_id: Any) -> Any:
    print(f"Loading the model {model_id}")

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                 pad_token_id=tokenizer.eos_token_id
                                                )
    
    model.to(device)

    llama_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        max_new_tokens=512,
        temperature=0.7,
        task="text-generation",  # LLM task
        torch_dtype=torch.float16,
        device_map="auto",
    )
    llm = HuggingFacePipeline(pipeline=llama_pipeline)
    return llm

## Setting the embeddings
def set_embeddings(model_name: Any) -> Any:
     print(f"\n------------------------------------")
     print(f"Setting the embeddings")
     embeddings = HuggingFaceEmbeddings(model_name=model_name)
     print(f"Embeddings set successfully")
     print(f"------------------------------------\n")
     return embeddings

## Loading the documents
def load_documents(local_directory_path: str) -> Any:
    # For PDF files
    print(f"\n------------------------------------")
    print(f"Loading PDFs from {local_directory_path}")
    loader = DirectoryLoader(local_directory_path,
                                glob='*.pdf',
                                loader_cls=PyPDFLoader)
    print(loader)
    documents = loader.load()
    print(f"Documents Loaded")
    print(f"------------------------------------\n")
    return documents


## Processing the documents
def process_documents(documents: Any) -> Any:
    print(f"\n------------------------------------")
    print(f"Processing the documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                    chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Documents processed")
    print(f"------------------------------------\n")
    return texts


## Saving the document to FAISS DB
def save_to_vectorstore(texts: Any, embeddings: Any, vectorestore_path: str) -> Any:
     print(f"\n------------------------------------")
     print(f"Saving the vectorestore to {vectorestore_path}")
     vectorstore = FAISS.from_documents(texts, embeddings)
     vectorstore.save_local(vectorestore_path)
     print(f"Vectore DB stored at {vectorestore_path}")
     print(f"------------------------------------\n")
     return vectorstore

## Setting the custom prompt
def set_custom_prompt() -> Any:
    """
    Prompt template for QA retrieval for each vectorstore
    """
    print("Setting the custom prompt")
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


## Setting up retrieval qa chain
def retrieval_qa_chain(llm: Any, vectorstore: Any) -> Any:
     chain = RetrievalQA.from_chain_type(llm=llm,
                                         chain_type='stuff',
                                         retriever=vectorstore.as_retriever(search_kwargs={'k': 2}), 
                                         return_source_documents=True,
                                         chain_type_kwargs={'prompt': set_custom_prompt()},
                                         verbose=False
                                         )
     return chain


## Defining the chatbot logic
def chatbot(llm: Any, vectorstore: Any) -> Any:
    chain = retrieval_qa_chain(llm, vectorstore)
    exit_conditions = ("exit", "bye", "quit", ":q")
    while True:
        user_input = input("User: ")
        
        if user_input.lower() in exit_conditions:
            print("Chatbot: Thanks!")
            break
        result = chain({"query": user_input})

        response = result["result"]
        
        print("Chatbot: ", response)
        print("--------------------------------------------------\n\n")


if __name__ == "__main__":
    documents = load_documents("/home/sagemaker-user/content/corpus")
    texts = process_documents(documents)
    embeddings = set_embeddings('sentence-transformers/all-mpnet-base-v2')
    vectorstore = save_to_vectorstore(texts, embeddings, "/home/sagemaker-user/content/vectorstore/")
    chatbot(load_llama("meta-llama/Llama-2-7b-chat-hf"), vectorstore)