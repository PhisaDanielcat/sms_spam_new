from transformers import AutoTokenizer, AutoModel , AutoModelForSequenceClassification

model_name = 'models/bert-tiny-finetuned-sms-spam-detection'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

raw_inputs = [
    "Ok lar... Joking wif u oni...."
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)

print(outputs[0][0][0])