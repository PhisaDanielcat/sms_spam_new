import numpy as np

def softmax(vector):
    # 防止溢出，先减去最大值
    e_x = np.exp(vector - np.max(vector))
    return e_x / e_x.sum()

# 测试
vector = np.array([1.2009, -1.5044])
softmax_output = softmax(vector)
print(softmax_output)
