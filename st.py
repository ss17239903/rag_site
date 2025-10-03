import streamlit as st
#from rag_model import *
from agentic_rag import *
from utils import *
from db_init import *
from summary import *
st.title("meow")
db_url = st.secrets["DATABASE_URL"]
# db_url = "postgresql://neondb_owner:npg_vqebF6uQH5Nt@ep-young-cherry-a8vnq6jm-pooler.eastus2.azure.neon.tech/neondb?sslmode=require&channel_binding=require"
memory = PostgresChatMemory(db_url)

if 'thread_id' not in st.session_state:
    print("generating thread id")
    st.session_state.thread_id = generate_thread_id()
    print("thread_id: ", st.session_state.thread_id)

thread_id = st.session_state.thread_id

with st.form('user_input'):
    user = st.text_input("enter your username")
    submit = st.form_submit_button("submit")

if submit:
    exist = memory.check_user(user)
    st.session_state.user_id = memory.create_user(user)

    if exist:
        st.write("welcome back, ", user, "!")
        st.session_state.existing_user = True
    else:
        st.write("created an account for you 🥳 ")
        st.session_state.existing_user = False

if 'user_id' in st.session_state:
    user_id = st.session_state.user_id

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # load user messages
        if st.session_state.existing_user:
            thread_msgs, msgs = memory.load_user(user_id)
            # if there's less than 10 total messages, just print them all
            if len(msgs) > 10:
                st.session_state.messages.append({"role": "assistant", "content": "Summarizing our message history..."})
                summaries = memory.load_summary(user_id)
                if summaries is None:
                    summaries = defaultdict(list)
                # for each thread in the user's history...
                for thread in thread_msgs:
                    print("thread: ", str(thread))
                    # ...summarize if it hasn't been summarized already...
                    if summaries[thread] == "":
                        summary = summarize_text(thread_msgs[thread])
                        print("summary: ", summary)
                        memory.save_summary(user_id, thread, summary)
                        summaries[thread] = summary
                    #...then add summary to message history.
                    st.session_state.messages.append({"role": "assistant", "content": summaries[thread]})
            else:
                st.session_state.messages = msgs

    graph = build_graph()

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
        memory.save(thread_id, user_id, role="user", content=prompt)

        # display user input
        with st.chat_message("user", avatar = "👾"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar = "👻"):
            # pass in thread_id to langchain model
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            print("current config: ", config)
            msgs = st.session_state.messages
            print(msgs)
            for msg in msgs:
                print(msg["role"])
            # get response from model
            response = graph.invoke({"messages": msgs}, config = config)
            #display model response
            msg = response["messages"][-1].content
            st.markdown(msg)
            st.session_state.messages.append({"role":"assistant", "content": msg})
            memory.save(thread_id, user_id, role="assistant", content=msg)
