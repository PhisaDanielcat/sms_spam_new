import pandas as pd
import tiktoken
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset

from TransformerClassifier_model import TransformerClassifier
from SpamDataset_class import  SpamDataset

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = SpamDataset(
    csv_file="datasets/train.csv",
    max_length=None,
    tokenizer=tokenizer
)
val_dataset = SpamDataset(
    csv_file="datasets/validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="datasets/test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

num_workers = 0
batch_size = 8

# torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)


# Instantiate the model with the appropriate input size
vocab_size = 50257  # The tokenizer size
model = TransformerClassifier(vocab_size=vocab_size)

# Print model architecture
print(model)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("training device is",device)


def train_epoch(model, data_loader, criterion, optimizer, device=device):
    model.to(device)  # 将模型移到指定的设备（GPU）
    model.train()  # 设置模型为训练模式

    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for input_ids, labels in data_loader:
        # 将数据也移到指定的设备（GPU）
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(input_ids)
        loss = criterion(output, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


# Example training loop over epochs
epochs = 8
for epoch in range(epochs):
    train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")


# Evaluation loop
def evaluate(model, data_loader, device=device):
    # 将模型移到指定设备
    model.to(device)
    model.eval()

    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            # 将数据移到指定设备
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            output = model(input_ids)
            loss = criterion(output, labels)

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

# Evaluate on validation set
val_loss, val_accuracy = evaluate(model, val_loader)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

save_dir = './models/my_trans.pth'
torch.save(model.state_dict(),save_dir)
