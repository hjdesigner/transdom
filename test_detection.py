import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "papluca/xlm-roberta-base-language-detection"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def detect_language(text: str):
  inputs = tokenizer(text, return_tensors="pt", truncation=True)

  # No gradient tracking needed - we're not traning, just predicting
  with torch.no_grad():
    outputs = model(**inputs)

    # Raw scores per language (logits) -> probabilities that sum to 100%.
    probabilities = torch.nn.functional.softmax(outputs.logits, dim =-1)[0]

    best_index = torch.argmax(probabilities).item()
    confidence = probabilities[best_index].item()

    # The model config maps class index (0, 1, 2...) back to a language code.
    language_code = model.config.id2label[best_index]

    return language_code, confidence

test_sentences = [
  "Welcome to our website",
  "Bem-vindo ao nosso site",
  "Bienvenido a nuestro sitio web",
  "Login",
  "Home",
  "hey",
]

for sentence in test_sentences:
  lang, confidence = detect_language(sentence)
  print(f"{sentence!r:35} -> {lang} ({confidence:.2%})")