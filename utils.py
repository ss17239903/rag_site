from langchain_milvus import Milvus
from pymilvus import MilvusClient
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

import os
import uuid
import pickle


load_dotenv()

def generate_thread_id():
    return str(uuid.uuid4())

def init_llm():
    llm = ChatOpenAI(
    model="gpt-5-nano-2025-08-07",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2)
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
    # COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
    # llm = init_chat_model("command-a-03-2025", model_provider="cohere")
    # GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    # llm = ChatGoogleGenerativeAI(
    #     model = "gemini-2.5-flash",
    #     temperature = 0,
    #     max_tokens = None,
    #     timeout = None,
    #     max_retries = 0,
    # )

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
