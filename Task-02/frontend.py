import streamlit as st
from RAG import rag

def clear_chat_history():
    st.session_state.messages = []

st.title("RAG Chat Bot 🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = rag(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.title("Chat History")
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.sidebar.markdown(f"{message['content']}")

st.sidebar.button("Clear Chat History", on_click=clear_chat_history)