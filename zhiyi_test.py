from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import streamlit as st
import random
import torch
import time

st.title("Mood Detection")

dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
classifier = pipeline("text-classification",
                        model='bhadresh-savani/bert-base-uncased-emotion', 
                        return_all_scores=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "emotion_states" not in st.session_state:
    st.session_state["emotion_states"] = []

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
    prediction = classifier(prompt, )
    
    return get_highest_emotion(prediction[0])[0]
    
def generate_response_with_context(prompt, chat_history, dialogpt_tokenizer, dialogpt_model):
    # Format chat history
    context = prompt
    for i in chat_history:
        context += " "
        context += i
    
    print(context)
    # Tokenize input
    inputs = dialogpt_tokenizer.encode(context, return_tensors="pt")
    
    # Generate response
    response_ids = dialogpt_model.generate(
        inputs,
        max_length=200,
        pad_token_id=dialogpt_tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    
    # Decode and handle empty responses
    response = dialogpt_tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    if not response.strip():
        response = "I couldn't generate a response. Can you clarify?"

    return response


def generate_response(prompt, chat_history_ids=None):
    inputs = dialogpt_tokenizer.encode(prompt, return_tensors="pt")
    if chat_history_ids is not None:
        inputs = torch.cat([chat_history_ids, inputs], dim=-1)
    response_ids = dialogpt_model.generate(inputs, max_length=100, pad_token_id=dialogpt_tokenizer.eos_token_id)
    response = dialogpt_tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response, response_ids

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        print("prompt: ", prompt)
        emotion = run_bert_eval(prompt)
        st.session_state["emotion_states"].append(emotion)
        response = generate_response_with_context(prompt, st.session_state["emotion_states"], dialogpt_tokenizer, dialogpt_model)
        print("response: ")
        response = st.write(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})