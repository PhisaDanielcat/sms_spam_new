import torch.nn as nn
import torch.optim as optim
import torch
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size=256, num_heads=2, num_classes=2, num_layers=1, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        # Embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # Transformer encoder layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,  # Dimension of input and output
            nhead=num_heads,  # Number of heads in multiheadattention
            dropout=dropout,  # Dropout for regularization
            dim_feedforward = 256
        )

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

        # Final classifier layer
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)

        # Pass through transformer encoder
        # Transformer expects input shape [seq_len, batch_size, emb_size]
        transformer_out = self.transformer_encoder(embedded.permute(1, 0, 2))  # [batch_size, seq_len, emb_size]

        # Pooling: Use the last hidden state for classification
        pooled_output = transformer_out.mean(dim=0)  # Mean pooling across the sequence length

        # Classifier output
        output = self.fc(pooled_output)

        return output

if __name__ == "__main__":
    vocab_size = 50257  # The tokenizer size
    model = TransformerClassifier(vocab_size=vocab_size)
    print(model)
