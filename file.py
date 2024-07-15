import os
import requests
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Load your model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pinecone setup
api_key = "8826f026-a047-467a-959c-31e9344822ab"
pc = Pinecone(api_key=api_key)
index_name = 'myindex'

# Check if the index exists
if index_name in pc.list_indexes().names():
    index = pc.Index(index_name)
else:
    raise ValueError(f"Index '{index_name}' does not exist in Pinecone.")

# Function to query Pinecone
def query_pinecone(query, top_k=3):
    query_vector = model.encode(query).tolist()
    response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return response

# Function to query the local LLaMA 3 model server
def query_llama_3(prompt, server_url):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}"  # Adjust if authentication is needed
    }
    payload = {
        'model': "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
        'messages': [
            {"role": "system", "content": "Provide insights based on the Thirukkural translations."},
            {"role": "user", "content": prompt}
        ],
        'temperature': 0.7
    }
    response = requests.post(f"{server_url}/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': response.text}

# Example query
user_query = "I feel sad"

# Query Pinecone
pinecone_results = query_pinecone(user_query)

# Extract relevant metadata from Pinecone results
metadata = [result['metadata'] for result in pinecone_results['matches']]
#print("Metadata Structure:", metadata)  # Uncomment this line to print the metadata

# Extract mk, mv, and sp from metadata
context = " ".join([f"mk: {meta.get('mk', 'unknown')}, mv: {meta.get('mv', 'unknown')}, sp: {meta.get('sp', 'No details available')}" for meta in metadata])
print("CONTEXT")
print(context)
print("RESPONSE")

# Construct the prompt using the user query and context
llama_3_prompt = f"You are an empathetic psychologist, also an author of a book called Thirukkural. Your name is Thiruvalluvar. You will be now advising a person based on your couplets. For each person, based on their problem and relevant couplets, your job is to convey them empathetically the teachings in your couplets with relevant quotes. You are living in Tamil Nadu. Give your responses like a Tamil boy talking on social media (Tanglish). Example response: idhuku edhuku da kavala padre. Naan enna sonnen.. ? ## Problem: {user_query} ## Relevant couplets: {context}"

# Query LLaMA 3
llama_3_server_url = "http://localhost:1234/v1"  # Adjust the URL to your LM Studio setup
llama_3_response = query_llama_3(llama_3_prompt, llama_3_server_url)

# Print response
print("LLaMA 3 Response:", llama_3_response)
