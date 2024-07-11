import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()
        # shape of pe: [1, max_length, d_model]

        pe = torch.zeros(max_length, d_model)   # shape: [max_length, d_model]
        position = torch.arange(0,max_length).unsqueeze(1)  # shape: [max_length, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) 
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # shape: [1, max_length, d_model]
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # pe[:, :x.size(1)] shape: [1, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x # Shape remains [batch_size, seq_len, d_model]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        # Q, K, V  
        scores = torch.matmul(Q, K.transpose(-2,-1))
        scaled_scores = scores / (self.d_k ** 0.5)
        attention_scores = torch.softmax(scaled_scores,dim=-1)
        output = torch.matmul(attention_scores, V) # shape: [batch_size, num_heads, seq_len, d_k]

        return output   # shape: [batch_size, num_heads, seq_len, d_k]
    
    def forward(self, X): 
        # X.shape: [batch_size, seq_len, d_model]
        batch_size = X.size(0)

        Q = self.W_q(X) # shape: [batch_size, seq_len, d_model]
        K = self.W_k(X) # shape: [batch_size, seq_len, d_model]
        V = self.W_v(X) # shape: [batch_size, seq_len, d_model]

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2) # shape: [batch_size, num_heads, seq_len, d_k]
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2) # shape: [batch_size, num_heads, seq_len, d_k]
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2) # shape: [batch_size, num_heads, seq_len, d_k]

        output = self.scaled_dot_product_attention(Q, K, V) # shape: [batch_size, num_heads, seq_len, d_k]
        output = output.transpose(1,2).reshape(batch_size, -1, self.num_heads * self.d_k) # shape: [batch_size, seq_len, num_heads * d_k]
        output = self.W_o(output) # shape: [batch_size, seq_len, d_model]

        return output   # shape: [batch_size, seq_len, d_model]

class PointwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = self.fc2(self.relu(self.fc1(x)))    # shape: [batch_size, seq_len, d_model]
        return x    # shape: [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PointwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        attn_output = self.self_attn(x) # shape: [batch_size, seq_len, d_model]
        x = self.norm1(x + attn_output) # shape: [batch_size, seq_len, d_model]
        ff_output = self.feed_forward(x) # shape: [batch_size, seq_len, d_model]
        x = self.norm2(x + ff_output) # shape: [batch_size, seq_len, d_model]

        return x    # shape: [batch_size, seq_len, d_model]


    