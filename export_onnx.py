from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.onnx import export
import torch

# 加载模型和分词器
model_name = 'models/bert-tiny-finetuned-sms-spam-detection'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()

# 准备示例输入文本
input_text = "This is a spam message example"

# 对输入文本进行分词
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# 定义模型的输入示例，准备导出
# 这里我们使用模型的输入来创建一个 dummy_input（假数据），以便 ONNX 能够进行推理
# 将输入数据准备为字典格式
dummy_input = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"]
}

# 导出到 ONNX
onnx_output_path = "model.onnx"

# 这里的 export() 方法需要根据你的模型来调整
torch.onnx.export(
    model, 
    (dummy_input["input_ids"], dummy_input["attention_mask"]),  # 输入是元组而不是字典
    onnx_output_path, 
    opset_version=14
)