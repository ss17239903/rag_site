import streamlit as st
#from rag_model import *
from agentic_rag import *
from utils import *

st.title("meow")

graph = build_graph()

# initialize chat history
if 'thread_id' not in st.session_state:
    print("generating thread id")
    st.session_state.thread_id = generate_thread_id()
    print("thread_id: ", st.session_state.thread_id)
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
    user_input = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_input)

    # display user input
    with st.chat_message("user", avatar = "👾"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar = "👻"):
        # pass in thread_id to langchain model
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        print("current config: ", config)
        # get response from model
        response = graph.invoke({"messages": [user_input]}, config = config)
        #display model response
        msg = response["messages"][-1].content
        st.markdown(msg)
        st.session_state.messages.append({"role":"assistant", "content": msg})
