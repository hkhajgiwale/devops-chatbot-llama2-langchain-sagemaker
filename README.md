
# devops-chatbot-llama2-langchain-sagemaker


We intend to create LLM based AI Chatbot that would answer the questions about Linux

It is simple LLM bot that answers the questions only on the trained dataset unlike RAG. We will train the data on the corpus of few Linux books like Linux Bible, etc. The vectorstore to store the learnings is FAISS database.


## Technologies



1. **LLM**: meta-llama/Llama-2-7b-chat-hf [https://huggingface.co/meta-llama/Llama-2-7b-chat-hf]. Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is developed at Meta and model is available at HuggingFace.

2. **VectorStore:** ___FAISS___ => FAISS (Facebook AI Similarity Search) is a library that allows developers to quickly search for embeddings of multimedia documents that are similar to each other [https://ai.meta.com/tools/faiss/]

3. **Embeddings:** ___sentence-transformers/all-mpnet-base-v2___ [https://huggingface.co/sentence-transformers/all-mpnet-base-v2]

4. **Langchain:** LangChain is a framework designed to simplify the creation of applications using large language models. We use it to load our documents, process them and use RetrievalQA library to return our results. Langchain Doc

5. **AWS Sagemaker:** Since the corpus used for training is humongous, we will be using ml.m5.8xlarge instance type for deploying Jupyter Notebooks.


## Data Corpus

Since we need to train our LLM to answer the questions pertaining the field of DevOps, we use a large corpus of data available at git repo Free-DevOps-Books at https://github.com/sreddy-bwi/Free-DevOps-Books-1. This repo contains the books in epub, pdf formats. We are only interested in pdf books. Additional to these books, we have also added Linux Bible and few more books.


## Steps of deployment

1. Login to AWS account and deploy a notebook in AWS Sagemaker
2. Login to HuggingFace account and create the personal access token
3. From HuggingFace account, send out an access request to access meta-llama/Llama-2-7b-chat-hf model
4. Login to huggingface programmatically using personal access token
5. Load the documents
6. Process the documents
7. Set the embeddings
8. Save the vector data to FAISS DB
9. Query the bot
