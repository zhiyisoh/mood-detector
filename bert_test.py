# from transformers import pipeline

# # Load the BERT-Emotions-Classifier
# classifier = pipeline("text-classification", model="ayoubkirouane/BERT-Emotions-Classifier")

# # Input text
# text = "Woohoo"

# # Perform emotion classification
# results = classifier(text)

# # Display the classification results
# print(results)

# This uses DistilBERT
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_name = "ahmettasdemir/distilbert-base-uncased-finetuned-emotion"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create the text classification pipeline
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

# Example text to classify
text = "What is wrong with you"

# Perform text classification
result = pipeline(text)

# Print the predicted label
predicted_label = result[0]['label']
print("Predicted Emotion:", predicted_label)
