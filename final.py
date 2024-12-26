import transformers
import logging
import torch
from messages import messages
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

# remove any model logging errors to have a smoother chatbot experience
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load the fine-tuned model and tokenizer for DistilBERT
# This uses fine-tuned DistilBERT for emotion classification
model_name = "ahmettasdemir/distilbert-base-uncased-finetuned-emotion"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create the text classification pipeline
encoder_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

# utilisation of Llama-3.3 for response generation
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Create the response generation pipeline using llama
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


cont = True

# while loop to have multi - turn conversations
while cont:
    # get user input
    text = input("user> ")

    # if user types exit, end the conversation
    if text == 'exit':
      cont = False
      break

    # parse the input to the encoder model (DistilBERT) to get the emotion class
    result = encoder_pipeline(text)
    predicted_label = result[0]['label']

    # add the new modified prompt into the message log.
    messages.append({"role": "user", "content": text + ". Reply me while considering that my current mood is " + predicted_label})

    # parse the message log into the decoder model (Llama 3.2) to generate human like text as a response
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )

    # get the output of the Llama model
    response = outputs[0]["generated_text"][-1]["content"]
    messages.append({"role": "system", "content": response})

    # print the reponse
    print("partner> ", response)

    # print message log
    # print("====================")
    # print(messages)
    # print("====================")