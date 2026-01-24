import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()
VECTORSTORE_PATH=r'data\vectorstore'
os.makedirs(VECTORSTORE_PATH, exist_ok=True)
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not found in environment")
