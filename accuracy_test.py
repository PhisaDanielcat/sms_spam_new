from transformers import AutoTokenizer, AutoModel , AutoModelForSequenceClassification
import pandas as pd
import time
import torch

model_name = 'models/bert-tiny-finetuned-sms-spam-detection'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


df = pd.read_parquet('datasets/sms_spam/plain_text/train-00000-of-00001.parquet')
sms_list = df['sms'].tolist()
label_list = df['label'].tolist()

correct_num = 0
total_num = 0
start_time = time.time()

for i in range(len(sms_list)):
    raw_inputs = sms_list[i]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    if predictions == label_list[i]:
        correct_num += 1
    total_num += 1
end_time = time.time()
print("total accuracy is %.2f"%(correct_num/total_num))
print("total time is %.2fs"%(end_time-start_time))
print("total item is %.2d"%(total_num))