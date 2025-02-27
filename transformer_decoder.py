
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Define input parameters
batch_size = 2  # Number of sequences processed in parallel
seq_len = 5  # Length of each sequence
d_model = 64  # Dimension of the embedding vectors
num_heads = 8  # Number of attention heads
d_ff = 256  # Dimension of the feed-forward layer
dropout = 0.1  # Dropout rate
vocab_size = 10000  # Vocabulary size
max_seq_len = 50  # Maximum sequence length

# Generate random target tensor and encoder output
tgt = torch.randint(0, vocab_size, (batch_size, seq_len))  # Random token indices
enc_out = torch.rand(batch_size, seq_len, d_model)  # Encoder output

# Create target mask for self-attention (lower triangular to prevent peeking into future tokens)
tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
memory_mask = None  # No mask for cross-attention

# Embedding and Positional Encoding
embedding = nn.Embedding(vocab_size, d_model)  # Word embedding layer
pos_embedding = nn.Embedding(max_seq_len, d_model)  # Positional encoding
positions = torch.arange(0, seq_len, device=tgt.device).unsqueeze(0)  # Generate position indices

# Embed the target input and add positional encoding
tgt_emb = embedding(tgt) + pos_embedding(positions)

# Function for Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)  # Get the dimension of the key
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # Compute scaled dot-product scores
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))  # Apply mask to ignore future tokens
    attn_probs = F.softmax(attn_scores, dim=-1)  # Normalize scores with softmax
    return torch.matmul(attn_probs, V)  # Multiply with values to get output

# Multi-Head Attention mechanism
def multi_head_attention(Q, K, V, d_model, num_heads, mask=None):
    d_k = d_model // num_heads  # Dimension per head
    
    # Linear layers to project Q, K, V
    W_q = nn.Linear(d_model, d_model)
    W_k = nn.Linear(d_model, d_model)
    W_v = nn.Linear(d_model, d_model)
    W_o = nn.Linear(d_model, d_model)  # Output projection layer
    
    # Transform Q, K, V into multiple heads and reshape
    Q = W_q(Q).view(batch_size, -1, num_heads, d_k).transpose(1, 2)
    K = W_k(K).view(batch_size, -1, num_heads, d_k).transpose(1, 2)
    V = W_v(V).view(batch_size, -1, num_heads, d_k).transpose(1, 2)
    
    # Compute scaled dot-product attention for each head
    attn_output = scaled_dot_product_attention(Q, K, V, mask)
    
    # Concatenate heads and project back to original dimension
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
    return W_o(attn_output)

# Feed-Forward Network
def feed_forward(x, d_model, d_ff):
    fc1 = nn.Linear(d_model, d_ff)  # First dense layer
    fc2 = nn.Linear(d_ff, d_model)  # Second dense layer
    return fc2(F.relu(fc1(x)))  # Apply ReLU activation and project back to d_model

# Define Layer Normalization and Dropout layers
norm1 = nn.LayerNorm(d_model)
norm2 = nn.LayerNorm(d_model)
norm3 = nn.LayerNorm(d_model)
drop = nn.Dropout(dropout)

# Step 1: Self-Attention on Target Sequence (Masked Multi-Head Attention)
tgt_attn = multi_head_attention(tgt_emb, tgt_emb, tgt_emb, d_model, num_heads, tgt_mask)
tgt_emb = norm1(tgt_emb + drop(tgt_attn))  # Apply residual connection and normalization

# Step 2: Cross-Attention with Encoder Output (Multi-Head Attention)
cross_attn = multi_head_attention(tgt_emb, enc_out, enc_out, d_model, num_heads, memory_mask)
tgt_emb = norm2(tgt_emb + drop(cross_attn))  # Apply residual connection and normalization

# Step 3: Feed-Forward Network
ffn_out = feed_forward(tgt_emb, d_model, d_ff)
tgt_emb = norm3(tgt_emb + drop(ffn_out))  # Apply residual connection and normalization

# Final Output
decoder_output = tgt_emb

print("Decoder Output Shape:", decoder_output.shape)  # Print final shape of decoder output
