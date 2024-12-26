import transformers
import torch
from messages import messages
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

# utilisation of Llama-3.3
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Load the fine-tuned model and tokenizer
# This uses fine-tuned DistilBERT
model_name = "ahmettasdemir/distilbert-base-uncased-finetuned-emotion"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create the text classification pipeline
encoder_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages.append({"role": "system", "content": "hello"})

while True:

    text = input("user> ")
    result = encoder_pipeline(text)
    predicted_label = result[0]['label']
    messages.append({"role": "user", "content": text + ". Reply me where my current mood is " + predicted_label})
    print(messages)

    outputs = pipeline(
        messages,
        max_new_tokens=64,
    )
    print("reached here")
    response = outputs[0]["generated_text"][-1]
    messages.append({"role": "system", "content": response})
    print(messages)

    print("partner> ", response)
