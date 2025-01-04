from optimum.intel.neural_compressor import INCQuantizer, INCQuantizationConfig
from transformers import AutoModelForSequenceClassification

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")

# 创建量化器
quantizer = INCQuantizer.from_pretrained(model)

# 创建量化配置对象
quantization_config = INCQuantizationConfig(
    approach="post_training_static_quant",  # 量化方法
    weight_precision="int8",  # 权重量化精度
    activation_precision="int8"  # 激活量化精度
)

# 执行量化
quantized_model = quantizer.quantize(
    quantization_config=quantization_config,  # 传递正确的配置对象
    save_directory="./quantized_model"  # 保存路径
)

print("量化后的模型已保存至: ./quantized_model")
