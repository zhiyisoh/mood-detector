from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import streamlit as st
import random
import torch
import time

st.title("Mood Detection")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def get_highest_emotion(emotion_scores):
    # Find the emotion with the maximum score
    highest_emotion = max(emotion_scores, key=lambda x: x["score"])
    return highest_emotion["label"], highest_emotion["score"]

def run_bert_eval(prompt):
    # this bert encoder model is finetuned on bert-base-uncased emotion dataset to detect emotion
    classifier = pipeline("text-classification",
                        model='bhadresh-savani/bert-base-uncased-emotion', 
                        return_all_scores=True)
    prediction = classifier(prompt, )
    
    return get_highest_emotion(prediction[0])[0]

def decoder():
    # DialoGPT model
    dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    
# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        print("prompt: ", prompt)
        response = st.write(run_bert_eval(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})