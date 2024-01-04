
# devops-chatbot-llama2-langchain-sagemaker


We intend to create LLM based AI Chatbot that would answer the DevOps realted questions.

It is simple LLM bot that answers the questions only on the trained dataset just like RAG. We will train the data on the corpus of few Linux books like Linux Bible, etc. The vectorstore to store the learnings is FAISS database.


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

## Output

```User:  What is cloud computing
Chatbot:  Cloud computing refers to the practice of deploying applications and services on the internet with a cloud provider.
--------------------------------------------------

User:  What is AWS
Chatbot:  AWS stands for Amazon Web Services. It is a cloud computing platform offered by Amazon that provides a range of services for computing, storage, networking, database, analytics, machine learning, and more.
--------------------------------------------------

User:  What are some of the major services in AWS
Chatbot:  Some of the major services in AWS include:

* Compute Services (EC2, Lambda, Elastic Beanstalk)
* Storage Services (S3, EBS, Elastic File System)
* Database Services (RDS, DynamoDB, Redshift)
* Security, Identity & Compliance Services (IAM, Cognito, Certificate Manager)
* Application Services (API Gateway, AppSync, CloudFront)
* Analytics Services (Redshift, QuickSight, Athena)
* Machine Learning Services (SageMaker, Comprehend, Rekognition)
* Networking & Connectivity Services (VPC, Elastic IP, Direct Connect)

Note: This list is not exhaustive and is based on my understanding of the question.
--------------------------------------------------

User:  What is Devops
Chatbot:  DevOps is a set of principles and practices that aims to bring together software developers and operations teams to collaborate and automate the software delivery process, with the goal of faster and more reliable end-to-end delivery of software systems to end customers.
--------------------------------------------------

User:  Tell me about CI/CD
Chatbot:  CI/CD stands for Continuous Integration/Continuous Deployment. It refers to the practice of integrating and testing code changes frequently, and automatically deploying those changes to production after they pass automated tests. The goal of CI/CD is to improve the efficiency and reliability of software delivery by reducing the time and effort required to move code from development to production.
--------------------------------------------------

User:  list prominent ci/cd tools
Chatbot:  Jenkins
Bamboo
GoCD
Team City
Electric Cloud

Please note that the list of prominent CI/CD tools is not exhaustive, and there are many other tools available in the market.
--------------------------------------------------

User:  10 important commands in linux
Chatbot:  
10 important commands in Linux are:

1. ls (list files)
2. cd (change directory)
3. pwd (print working directory)
4. mv (move or rename files)
5. rm (remove files)
6. cp (copy files)
7. mkdir (create directory)
8. rmdir (remove directory)
9. echo (print text to a file)
10. cat (concatenate and display files)

Note: These are some of the most commonly used commands in Linux, but there are many more available.
--------------------------------------------------

User:  What is the difference between sudo -i and sudo su -
Chatbot:  sudo -i and sudo su - are both used to run commands as a different user, but they have slightly different usage and behavior.

sudo -i allows you to run a command as a different user without a password, while sudo su - requires a password to switch users.

In general, it's recommended to use sudo -i when you need to run a command as a different user without a password, and to use sudo su - when you want to switch users and run a command with a password.
--------------------------------------------------

User:  what is iphone
Chatbot:  iPhone is a brand of smartphones developed by Apple Inc.
--------------------------------------------------

User:  Who is the prime minister of INdia
Chatbot:  I don't know the answer to that question. Sanjeev is a cloud and DevOps expert, not a political analyst.
--------------------------------------------------

User:  Tell me something about networking in cloud
Chatbot:  The cloud has had a significant impact on networking, as it has enabled the emergence of new networking models and technologies. For example, AWS and OpenStack provide networking services out of the box, such as VPCs, subnets, and security groups. These services have made networking a commodity much like infrastructure, allowing developers to consume networking resources on demand. Additionally, cloud networking has enabled the use of software-defined networking (SDN) and network functions virtualization (NFV) technologies, which can be used to programmatically control and automate network functions.
--------------------------------------------------

User:  Bye
Chatbot: Thanks!
```

## Authors

- Harsh Khajgiwale [@hkhajgiwale](https://www.github.com/hkhajgiwale)


## 🚀 About Me
I have been engaged in the professional capacity as a seasoned DevOps Engineer for last 6+ years. My area of expertise includes but not limited to AWS, Python, Linux, Terraform, CI/CD, etc. 

I hold a Masters Degree in Data Sciences and Engineering. In my free time, I play around the data and explore AI technologies.

