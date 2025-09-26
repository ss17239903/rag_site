import streamlit as st
#from rag_model import *
from agentic_rag import *
from utils import *

# Create and set a new event loop for the current thread
# loop = asyncio.new_event_loop()
# asyncio.set_event_loop(loop)

st.title("meow")

graph = build_graph()
# st.markdown("""
#         <style>
#         div[data-testid="stChatMessage"] {
#             background-color: #CCCEB7 !important;
#         }
#         </style>
#     """, unsafe_allow_html=True)

# st.markdown("""
# <style>
#     [class*="st-key-user"] {
#         background-color: #C9B1BD !important;
#     }
#
#     [class*="st-key-assistant"] {
#         background-color: #D79233;
#     }
#   </style>
#  """, unsafe_allow_html=True)
# def chat_message(name):
#     return st.container(key=f"{name}-{uuid.uuid4()}").chat_message(name=name)

# initialize chat history
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = generate_thread_id()
if 'messages' not in st.session_state:
    st.session_state.messages = []

# display historical chat messages if they exist
for message in st.session_state.messages:
    if message["role"] == "user":
        avatar = "👾"
    else:
        avatar = "👻"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"], unsafe_allow_html=True)

# get user input
if prompt := st.chat_input("ask me anything (about philosophy)"):
    # add user input to messsage history
    st.session_state.messages.append({"role": "user", "content": prompt})
    langchain_messages = st.session_state.messages
    # langchain_messages = {"messages": [{"role":"user", "content":prompt}]}
    # display user input
    with st.chat_message("user", avatar = "👾"):
        st.markdown(prompt)

    with st.chat_message("ai", avatar = "👻"):
        # pass in thread_id to langchain model
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        # get response from model
        response = graph.invoke(langchain_messages, config = config)
        #display model response
        msg = response["messages"][-1].content
        st.markdown(msg)
        st.session_state.messages.append({"role":"ai", "content": msg})

        # for step in graph.stream(
        #     {"messages": [{"role": "user", "content": prompt}]},
        #     stream_mode="values",
        #     config = config,
        # ):
        #     msg = step["messages"][-1]
        #     msg.pretty_print()
        #     # step_record = step
        #     # add model response to message history
        # st.markdown(msg.content)
        # st.session_state.messages.append({"role":"assistant", "content": msg.content})
        #
        # try:
        #     context = step_record["context"]
        # except:
        #     context = []
        #     pass
        # sources = []
        # for doc in context:
        #     source = doc.metadata["comment_url"]
        #     if source is not None:
        #         sources.append(source)
        # st.markdown(sources)
        # st.session_state.messages.append({"role":"assistant", "content": sources})
