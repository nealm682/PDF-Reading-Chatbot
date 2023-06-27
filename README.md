# PDF Knowledge Base Using Vectors

This repository contains a simple yet powerful Python script to convert a PDF document into a searchable knowledge base. It uses the language understanding capabilities of OpenAI's GPT2 model, the document handling of PyPDF and Langchain, and FAISS for efficient similarity search. The output is a conversational agent that you can query for information contained in the PDF.

## Installation

Use pip to install the required packages:

```bash
pip install -q langchain==0.0.150 pypdf pandas matplotlib tiktoken textract transformers openai faiss-cpu streamlit
```

## Usage

First, set your OpenAI API key:

```python
import os
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
```

This script allows for PDF parsing in two ways:

- **Simple method (Split by pages)**

```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("path_to_your_pdf")
pages = loader.load_and_split()
```

- **Advanced method (Split by chunks)**

This method involves converting the PDF to text, counting tokens in the text, and splitting the text into manageable chunks:

```python
import textract
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter

doc = textract.process("path_to_your_pdf")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Function to count tokens
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Create function to count tokens
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([doc])
```

The code then uses FAISS to create a vector database from these chunks, and builds a question-answering system from this vector database and the language model. 

The script ends with a simple text-based interface for querying the system:

```python
print("Ask me a question about the PDF document you just uploaded. Type 'exit' to stop.")
input_box = widgets.Text(placeholder='Please enter your question here:')
input_box.on_submit(on_submit)
display(input_box)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)