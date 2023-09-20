import os

import dotenv
from langchain.embeddings import HuggingFaceEmbeddings

dotenv.load_dotenv()

CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")

USER_ID = "cohere"
APP_ID = "embed"
MODEL_ID = "cohere-text-to-embeddings"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
