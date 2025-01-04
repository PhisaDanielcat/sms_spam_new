from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# 加载模型和数据集
model_name = 'models/bert-tiny-finetuned-sms-spam-detection'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载数据集
dataset = load_dataset("datasets/sms_spam")

# 数据预处理函数：填充到最大长度
def preprocess_function(examples):
    return tokenizer(
        examples['sms'], 
        padding="max_length",  # 填充到最大长度
        truncation=True,       # 截断超出最大长度的部分
        max_length=128,        # 设置最大长度为 128
        return_tensors="pt"
    )

# 编码数据集
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 定义计算精度的函数
def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)  # 获取预测结果的类别索引
    labels = p.label_ids  # 获取真实标签
    accuracy = accuracy_score(labels, preds)  # 计算精度
    return {"accuracy": accuracy}

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # 每个epoch评估一次
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=15,
    weight_decay=0.01,
)

# 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["train"],
    compute_metrics=compute_metrics,  # 添加自定义的评估函数
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model("./fine_tuned_model")