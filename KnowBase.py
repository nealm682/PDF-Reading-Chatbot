# PDF Knowledge Base using vectors


# pip install packages
#!pip install -q langchain==0.0.150 pypdf pandas matplotlib tiktoken textract transformers openai faiss-cpu streamlit


# import packages
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain


# Set api key   
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"


# You MUST add your PDF to local files in order to use this method
# Simple method - Split by pages 
loader = PyPDFLoader("./SMILES-Related.pdf")
pages = loader.load_and_split()
# print(pages[0])

# Split by paragraphs
# chunks = pages


# Advanced method - Split by chunk
# Step 1: Convert PDF to text
import textract
doc = textract.process("./SMILES-Related.pdf")

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('SMILES-Related-2.txt', 'w') as f:
    f.write(doc.decode('utf-8'))

with open('SMILES-Related-2.txt', 'r') as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])