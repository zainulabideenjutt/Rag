import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
import os

OLLAMA_URL = "http://localhost:11434/v1"

# Local embeddings via Ollama's OpenAI-compatible endpoint
local_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="ollama",                 # any non-empty string works
    api_base=OLLAMA_URL,
    model_name="nomic-embed-text",
)

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(
    name="document_qa_collection_local",   # NEW name, see note below
    embedding_function=local_ef,
)

# Local chat model
client = OpenAI(api_key="ollama", base_url=OLLAMA_URL)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain RAG in two sentences."},
    ],
)
print(response.choices[0].message.content)

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents")


chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    for i, chunk in enumerate(chunks):
        chunked_documents.append(
            {
                "id": f"{doc['id']}_chunk_{i+1}",
                "text": chunk,
            }
        )
# chunked_documents = [{id , chunk}, {id, chunk}, ...]


def get_openai_embedding(text):
    response = client.embeddings.create(input=text, model="nomic-embed-text")
    embedding = response.data[0].embedding
    print("==== Generating embeddings... ====")
    return embedding

# Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_openai_embedding(doc["text"]) # adds embedding to each chunked document

# chunked_documents = [{id, chunk, embedding}, {id, chunk, embedding}, ...] 


# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )

# in database we will have collection like this : 
# [{id, chunk, embedding}, {id, chunk, embedding}, ...] 

def query_documents(question, n_results=2):
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=question, n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = []
    for sublist in results["documents"]:   # loop over each question's results
        for doc in sublist:                # loop over each chunk in that result
            relevant_chunks.append(doc)
    print("==== Returning relevant chunks ====")
    return relevant_chunks # returns [chunk1, chunk2, ...] of relevant chunks

def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context
    )

    response = client.chat.completions.create(
        model="llama3.2:3b",            
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content   


question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)