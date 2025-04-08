#Utilizes the data in the vector embedding database and combines the results with the GPT-4-turbo model, to ensure uptodate results
import os
from pinecone import Pinecone
import openai
from openai import OpenAI
import time

# Initialize OpenAI and Pinecone
client = OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Create a Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Connect to your Pinecone index
index_name = "repo-rag"
index = pc.Index(index_name)

def retrieve_relevant_chunks(query, top_k=20):
    """Embeds the query, retrieves relevant chunks from Pinecone, and returns them."""
    
    # Embed the user query
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
        encoding_format="float"
    )
    query_embedding = response.data[0].embedding

    # Search in Pinecone
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Extract relevant text chunks
    retrieved_chunks = [match['metadata']['text'] for match in search_results['matches']]
    
    return retrieved_chunks
def generate_answer(query):
    """Retrieves context from Pinecone and generates an answer using GPT."""
    
    retrieved_chunks = retrieve_relevant_chunks(query)
    
    # Create a formatted context for GPT
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""You are an AI assistant that answers questions based on the Mesa Repository, frame your answers based on the context given below.
                If you don't find information about a topic in the cotext, then don't display it as it maybe outdated and the context only contains the latest information. Please ensure you dont make use of any information not there in the context. 

    Context:
    {context}

    Query: {query}
    Answer:
    """
    #print(prompt)
    # Generate response using OpenAI's GPT
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

# Example usage
user_query = "what is agent set in mesa? "
answer = generate_answer(user_query)
print("AI Response:", answer)

