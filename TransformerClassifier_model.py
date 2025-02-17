import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# 自定义点积注意力函数
def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights

# 自定义 LayerNorm 函数
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# 自定义 MultiHeadAttention 模块
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)


    def forward(self, q, k, v, mask=None):
        residual = q

        # Linear projections
        q = self.w_q(q).view(q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(k.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(v.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        attn_output = attn_output.transpose(1, 2).contiguous().view(q.size(0), -1, self.d_model)

        # Output projection
        output = self.w_o(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

# 自定义 FeedForward 模块
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.0):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

# 自定义 TransformerEncoderLayer 模块
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, dropout=dropout)

    def forward(self, x):
        x = self.self_attn(x, x, x)
        x = self.feed_forward(x)
        return x

# 自定义 TransformerEncoder 模块
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 主模型
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size=256, num_heads=2, num_classes=2, num_layers=1, dropout=0.0):
        super(TransformerClassifier, self).__init__()

        # 使用容器组织网络层
        self.layers = nn.ModuleList([
            nn.Embedding(vocab_size, emb_size),  # Embedding layer
            TransformerEncoder(TransformerEncoderLayer(emb_size, num_heads, dropout), num_layers),  # Encoder
            nn.Linear(emb_size, num_classes)  # Classifier
        ])

    def forward(self, x):
        # Embedding layer
        embedded = self.layers[0](x)

        # Transformer encoder
        transformer_out = self.layers[1](embedded.permute(1, 0, 2))  # [seq_len, batch_size, emb_size]
        # Pooling: Mean pooling across the sequence length
        pooled_output = transformer_out.mean(dim=0)  # [batch_size, emb_size]
        # Classifier output
        output = self.layers[2](pooled_output)
        return output

if __name__ == "__main__":
    vocab_size = 50257  # The tokenizer size
    model = TransformerClassifier(vocab_size=vocab_size)
    print(model)
    model.load_state_dict(torch.load("models/my_trans.pth"))
    model.eval()

    dummy_input = torch.tensor(np.array([[25314, 248, 1329, 1122, 37882, 31640, 22367, 1942, 19061, 10507]]),dtype=torch.long)

    # Perform a forward pass through the model
    output = model(dummy_input)
    print(output)