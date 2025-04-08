#code to generate the vector embeddings in python of the docs and mesa sub dirs
import os
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm
import openai
from openai import OpenAI
client = OpenAI()
import time

# Load API keys from .env file
load_dotenv()

models = openai.models.list()

for model in models.data:
    print(model.id)
# Initialize OpenAI & Pinecone
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Create a Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Get or create index
index_name = "repo-rag"
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(index_name, dimension=1536, metric="cosine")

index = pc.Index(index_name)

# Allowed file types & directories to ignore
ALLOWED_EXTENSIONS = {".md", ".txt", ".py", ".ipynb",".yaml",".yml", ".ini"}

def extract_and_vectorize(repo_path):
    total_files = 0
    file_paths = []

    # Walk through the repo and filter only the desired subdirectories
    for root, dirs, files in os.walk(repo_path):
        # Ensure we only process files inside `docs/` or `mesa/`
        if not (root.startswith(os.path.join(repo_path, "docs")) or 
                root.startswith(os.path.join(repo_path, "mesa"))):
            continue  # Skip directories that are not `docs/` or `mesa/`

        for file in files:
            if os.path.splitext(file)[1] in ALLOWED_EXTENSIONS:
                file_paths.append(os.path.join(root, file))
                total_files += 1

    # Process files with tqdm
    for file_path in tqdm(file_paths, total=total_files, desc="Processing Files"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Chunking (every 1000 characters)
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]

            # Process chunks
            for chunk_id, chunk in tqdm(enumerate(chunks), total=len(chunks), 
                                        desc=f"Processing Chunks of {file_path}", leave=False):
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk,
                    encoding_format="float"
                )
                embedding = response.data[0].embedding

                # Store in Pinecone
                vector_id = f"{file_path}_{chunk_id}"
                index.upsert(vectors=[(vector_id, embedding, {"file": file_path, "text": chunk})])
                #time.sleep(2)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("All embeddings stored successfully in Pinecone!")

# Run the function
extract_and_vectorize("path where the github repo is cloned") # Replace with your repo path
