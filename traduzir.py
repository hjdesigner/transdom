from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"

# Load tokenizer and model separately (more control, more stable across versions)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# >>por<< = European Portuguese | >>pob<< = Brazilian Portuguese
text = ">>por<< The user clicked the button and the page updated instantly."

# Tokenize: covert text into numbers the model undestands
inputs = tokenizer(text, return_tensors="pt")

# Generate the translation
outputs = model.generate(**inputs)

# Decode: convert output tokens back into readable text
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translation)