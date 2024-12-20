from transformers import pipeline
import streamlit as st
import random
import time

st.title("Mood Detection")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def run_bert_eval(prompt):
    # this bert encoder model is finetuned on bert-base-uncased emotion dataset to detect emotion
    classifier = pipeline("text-classification",
                        model='bhadresh-savani/bert-base-uncased-emotion', 
                        return_all_scores=True)
    prediction = classifier(prompt, )
    return prediction

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        print("prompt: ", prompt)
        response = st.write_stream(run_bert_eval(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})