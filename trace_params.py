import torch
from TransformerClassifier_model import TransformerClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=6)
torch.set_printoptions(threshold=np.inf, linewidth=1e9, precision=6)
import os

def print_params(name):
    def hook(module, input, output):
        # 创建 params 目录（如果不存在）
        os.makedirs('params', exist_ok=True)

        # 将 name 中的 '/' 替换为 '_'，以避免路径问题
        filename = f'params/{name.replace(".", "_")}.txt'

        with open(filename, 'w') as f:
            f.write(f"Layer: {name}\n")
            f.write("Input:\n")
            f.write(f"{input[0].detach().cpu().numpy()}\n")  # 打印输入
            f.write("Output:\n")
            f.write(f"{output.detach().cpu().numpy()}\n")  # 打印输出

            # 打印模块的参数
            for param_name, param in module.named_parameters():
                f.write(f"Parameters of {param_name}:\n")
                f.write(f"{param}\n")

    return hook


def register_hooks(model):
    for name, module in model.named_modules():
        print(name)
        module.register_forward_hook(print_params(name))


# def print_params(name, module, input, output):
#     # 使用传入的 `name` 作为文件名
#     with open(f'params/{name}.txt', 'w') as f:
#         f.write(f"Layer: {module.__class__.__name__} ({name})\n")  # 加上 name
#         f.write("Input:\n")
#         f.write(f"{input[0].detach().cpu().numpy()}\n")  # Print the input
#         f.write("Output:\n")
#         f.write(f"{output.detach().cpu().numpy()}\n")  # Print the output
#
#         for name, param in module.named_parameters():
#             f.write(f"Parameters of {name}:\n")
#             f.write(f"{param}\n")  # 可以打印参数，也可以选择打印其他内容
#
# def register_hooks(model):
#     for name, module in model.named_modules():
#         print(name)
#         module.register_forward_hook(lambda module, input, output: print_params(name, module, input, output))





if __name__ == "__main__":
    vocab_size = 50257  # The tokenizer size
    model = TransformerClassifier(vocab_size=vocab_size)
    model.load_state_dict(torch.load("models/my_trans.pth"))
    model.eval()

    # Register hooks
    register_hooks(model)

    # Create a dummy input (batch size=2, sequence length=10)
    # dummy_input = torch.randint(0, vocab_size, (1, 10))
    dummy_input = torch.tensor(np.array([[25314,248,1329,1122,37882,31640,22367,1942,19061,10507]]),dtype=torch.long)

    # Perform a forward pass through the model
    output = model(dummy_input)

    print("Forward pass complete. Check params/*.txt for layer inputs, outputs, and parameters.")