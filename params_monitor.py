import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from TransformerClassifier_model import TransformerClassifier
import tiktoken
import pandas as pd

from SpamDataset_class import  SpamDataset

def get_model_params(model):
    total_params = 0
    param_details = []

    for name, param in model.named_parameters():
        print(name)
        print(param.shape)
        param_count = param.numel()
        total_params += param_count
        param_details.append(f"{name}: {param_count} parameters")
        # print(total_params)

    return total_params, param_details


if __name__ == "__main__":
    vocab_size = 50257  # The tokenizer size
    model = TransformerClassifier(vocab_size=vocab_size)
    model.load_state_dict(torch.load("models/my_trans.pth"))
    model.eval()
    print(model)

    get_model_params(model)