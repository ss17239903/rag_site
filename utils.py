
from pymilvus import MilvusClient
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from sentence_transformers import SentenceTransformer
from langgraph.checkpoint.postgres import PostgresSaver

import os
import uuid
import pickle
import random
import string


load_dotenv()

def generate_thread_id():
    return str(uuid.uuid4())

def generate_user_id():
    length = 8
    user_id = ''.join(random.choices(string.ascii_letters, k=length))
    return user_id

def init_checkpointer():
    DB_URI = "postgresql://neondb_owner:npg_vqebF6uQH5Nt@ep-young-cherry-a8vnq6jm-pooler.eastus2.azure.neon.tech/neondb?sslmode=require&channel_binding=require"
    checkpointer = PostgresSaver.from_conn_string(DB_URI)
    return checkpointer

def init_llm():
    llm = init_chat_model("command-a-03-2025", model_provider="cohere")
    return llm

def init_client():
    MILVUS_API_KEY = os.environ.get("MILVUS_API_KEY")
    URI = "https://in03-898dd5f98e8e8ef.serverless.aws-eu-central-1.cloud.zilliz.com"
    client = MilvusClient(
        uri=URI,
        token=MILVUS_API_KEY
    )
    return client

def init_vector_store():
    MILVUS_API_KEY = os.environ.get("MILVUS_API_KEY")
    URI = "https://in03-898dd5f98e8e8ef.serverless.aws-eu-central-1.cloud.zilliz.com"

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Milvus(
        embeddings,
        connection_args={"uri": URI, "token": MILVUS_API_KEY, "db_name": "f-db"},
        collection_name="philosophy",
    )

    return vector_store

def init_web_search():
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    web_search = TavilySearch(max_results=2)
    return web_search

def init_graph():
    graph = pickle.load(open("cluster_info.pkl", "rb"))
    return graph

def init_embedder():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder
