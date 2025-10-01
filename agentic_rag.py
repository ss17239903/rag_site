#environment location: /Users/sandhyasivakumar/opt/anaconda3/envs/lg - use conda activate lg to see an environment that works

from typing_extensions import List, Annotated, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool

from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


import os
import uuid
import numpy as np

from utils import *

# llm = init_llm()
# vector_store = init_vector_store()
client = init_client()
web_search = init_web_search()
cluster_info = init_graph()
embedder = init_embedder()

# decider_model = init_llm()
# response_model = init_llm()
llm = init_chat_model("command-a-03-2025", model_provider="cohere")
decider_model = init_chat_model("command-a-03-2025", model_provider="cohere")
response_model = init_chat_model("command-a-03-2025", model_provider="cohere")
#search_model = init_chat_model("command-a-03-2025", model_provider="cohere")

class State(MessagesState):
    context: List[Document]

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """retrieve information related to a query."""
    print("retrieving docs")
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]

    # Find top matching topics by centroid similarity
    scores = {cid: np.dot(q_emb, info["center"]) for cid, info in cluster_info.items()}
    top_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]

    filter_str = "topic in ["
    for topic in top_topics:
            filter_str = filter_str + str(int(topic[0])) + ","
    filter_str = filter_str[:-1] + "]"

    # Retrieve docs
    docs = client.search(
        collection_name= "phil",
        data= [q_emb],
        limit=10,
        filter=filter_str,
        output_fields=["text", "metadata"]
    )
    serialized = "\n\n".join((f"Source: {doc['entity']['metadata']}\nContent: {doc['entity']['text']}") for doc in docs[0])

    retrieved_docs = [
       Document(
           page_content=doc["entity"]["text"],
           metadata=doc["entity"]["metadata"]
        )
        for doc in docs[0]
    ]

    return serialized, retrieved_docs


# @tool(response_format="content_and_artifact")
# def retrieve(query: str):
#     """retrieve information related to a query."""
#     retrieved_docs = vector_store.similarity_search(query, k=2)
#     serialized = "\n\n".join(
#     (f"Source: {doc.metadata}\nContent: {doc.page_content}")
#     for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs


@tool(response_format="content_and_artifact")
def search(query: str):
    """Use Tavily web search """
    print("doing web search")
    # --- Step 1: web search ---
    response = web_search.invoke(query)  # list of dicts: {"url":..., "content":...}
    results = response["results"]

    if not results:
        return "No results found.", []

    # --- Step 2: prepare raw snippets ---
    raw_snippets = "\n\n".join(
        f"Source: {doc.get('url','unknown')}\nContent: {doc.get('content','')}"
        for doc in results
    )

    retrieved_docs = [
        Document(
            page_content=doc.get("content", ""),
            metadata={"source": doc.get("url", "unknown")}
        )
        for doc in results
    ]

    return raw_snippets, retrieved_docs

def run_search(state:State):
    """call the model to run a web search for the user query. always generate a tool call for search."""
    print("running search")
    response = search(state["messages"][0].content)
    return {"messages": [response]}

def query_or_respond(state: State):
    """call the model to generate a response based on the current state. given the question, generate tool call for retrieval, otherwise respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


class SearchNecessity(BaseModel):
    """Decide if it would be helpful to run a web search to supplement the retrieved document context. """

    binary_score: str = Field(
        description = "A binary score representing whether or not a search should be run."
    )

SEARCH_NECESSITY_PROMPT = (
    "Decide whether a web search is necessary to answer the user's question, given the user's question and the retrieved document context."
    "Give a binary yes or no score to indicate whether a web search is necessary."
    "Only answer yes if the user's question relates to current events or is location-based (for example, 'philosophy classes near me')."
    "Answer no if the retrieved document context is sufficient to answer the user question."
    "If you do not have enough information to decide, answer no."
    "here is the user question: \n\n {question} \n\n"
    "here is the document context: {context} \n"
)

def search_necessity(state:State):
    """determine whether a web search is needed."""
    question = state["messages"][0].content
    recent_tool_messages = []

    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break

    tool_messages = recent_tool_messages[::-1]
    context = []

    for tool_message in tool_messages:
        context.extend(tool_message.artifact)

    prompt = SEARCH_NECESSITY_PROMPT.format(question=question, context=context)
    response = (
        decider_model.with_structured_output(SearchNecessity).invoke(
            [{"role":"user", "content":prompt}]
        )
    )

    score = response.binary_score
    return score

def generate(state: State):
    """generate answer."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break

    tool_messages = recent_tool_messages[::-1]
    context = []
    for tool_message in tool_messages:
        context.extend(tool_message.artifact)
    # docs_content = "\n\n".join(doc.content for doc in tool_messages)

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]


    # for i, doc in enumerate(docs, 1):
    #     sources.append(f"[{i}] {doc.metadata.get('url', 'No URL')}")
    # text_blobs = [f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)]
    # return "\n\n".join(text_blobs), "\n".join(sources)

    system_message_content = (
        "You are an interactive and transparent expert guide."
        "You have an ontology in your mind of the topic the user is asking about. Expose this to the user, ideally in the form of a diagram or chart."
        """Provide an answer using the following structure:
            1. a concise summary of your response
            2. a list of areas to explore
            3. a topic breakdown
            4. references
        """
        "If there are distinct differing perspectives on the topic, organize your answer by those viewpoints."

        "Do not assume anything beyond what the user has explicity stated. If you do want to make inferences, make sure to clarify with the user that they are correct before proceeding."
        "If the user's question is broad, give them options on how to narrow down their line of questioning. Do this at the beginning of your answer."
        "Use the following pieces of retrieved context to assist the user in understanding the topic they are interested in thoroughly."
        "\n\n"
        "{context}"
        "Make sure to answer the user's question using the vernacular of an expert in the field."

        "If you don't have the information to answer the question, let the user know."
        "Cite sources inline using [1], [2], etc. Do not invent sources. Only use those provided. At the end, include a References section with the numbered URLs."
    ).format(context=context)

    prompt = [SystemMessage(system_message_content)] + conversation_messages
    #structured_llm = llm.with_structured_output(AnswerWithSources)
    response = llm.invoke(prompt)

    return {"messages": [response], "context": context}

def build_graph():
    gb = StateGraph(State)

    gb.add_node("query_or_respond", query_or_respond)
    gb.add_node("retrieve", ToolNode([retrieve]))
    gb.add_node("run_search", run_search)
    gb.add_node("generate", generate)

    gb.set_entry_point("query_or_respond")
    gb.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END:END, "tools": "retrieve"}
    )

    gb.add_conditional_edges(
        "retrieve",
        search_necessity,
        {"yes":"run_search", "no":"generate"}
    )

    gb.add_edge("run_search", "generate")
    gb.add_edge("generate", END)

    graph = gb.compile()

    memory = InMemorySaver()
    graph = gb.compile(checkpointer=memory)

    return graph
#https://docs.langchain.com/langsmith/evaluation-approaches

#https://docs.langchain.com/langsmith/evaluation-approaches
#https://docs.langchain.com/langsmith/evaluation-approaches
#https://docs.langchain.com/langsmith/evaluation-approaches
#"what's an approrpirate stitch length for different types of fabric"
# init_llm()
# vector_store = init_vector_store()
#
# graph = build_graph()
#
# config = {"configurable": {"thread_id": "1"}}
# input_message = "what's the best sewing machine for beginners?"
# response = graph.invoke({"messages": [{"role": "user", "content": input_message}]}, config = config)
#
# response["messages"][-1].pretty_print()
#
# input_message2 = "where can i take classes on how to use a sewing machine?"
# response2 = graph.invoke({"messages": [{"role": "user", "content": input_message2}]}, config = config)
#
# response2["messages"][-1].pretty_print()

#agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)

# config = {"configurable": {"thread_id": "1"}}
#
# input_message = (
#     "what are some good sewing machines for beginners?"
#     "how do i use the machine you recommended?"
# )
#
# for event in agent_executor.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode = "values",
#     config = config,
# ):
#     event["messages"][-1].pretty_print()

# input_message = "how do i start learning how to sew?"
# for step in graph.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()
