import streamlit as st
from RAG import rag

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = []


# Main title of the app
st.title("RAG Chat Bot 🤖")

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate assistant response
    response = rag(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar to display chat history
st.sidebar.title("Chat History")
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.sidebar.markdown(f"{message['content']}")

# Set up the sidebar
st.sidebar.button("Clear Chat History", on_click=clear_chat_history)