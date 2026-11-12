!pip install -q langchain_community
!pip install -q langchain-text-splitters
!pip install -q langchain-huggingface
!pip install -q langchain-qdrant
!pip install -q langchain-groq
!pip install -q unstructured
!pip install -q python-docx
!pip install -q sentence-transformers

import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
file_path = "/content/CHURN PREDICTION ANALYSIS Questions.docx"
if not os.path.exists(file_path):
    print("File not found!")
    exit()
loader = UnstructuredFileLoader(file_path)
documents = loader.load()
print(f"\nLoaded {len(documents)} document(s).")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    location=":memory:",
    collection_name="documents"
)
print("Vector database created successfully!")
llm = ChatGroq(
    api_key="your_key"
)
print("\nTalk to your document!")
print("Type 'exit' to quit.\n")
while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break
    results = vector_store.similarity_search(question, k=3)
    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    prompt = f"""
You are an AI assistant.

Use ONLY the information given below.

If the answer is not available in the document, say:

"I couldn't find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nAssistant:")
    print(response.content)
    print("-" * 70)
