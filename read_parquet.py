import pandas as pd

# 读取单个 Parquet 文件
df = pd.read_parquet('datasets/sms_spam/plain_text/train-00000-of-00001.parquet')

# 查看数据
# print(df.head())
sms_list = df['sms'].tolist()
label_list = df['label'].tolist()

print(sms_list[0])  # 输出 sms 列前5个元素
print(label_list[0])  # 输出 label 列前5个元素