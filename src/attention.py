import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler as profiler

class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        # token -> embeding vector
        self.embedding_layer = nn.Embedding(10000, d_model)

        # X * weight 
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # final output layer
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, input_tokens):
        with profiler.record_function("attention Forward"):
            batch_size = input_tokens.shape[0]

            # 1. token -> embedding vector
            X_embedded = self.embedding_layer(input_tokens)  # 2*10*512

            # 2. Q,K,V 
            Q = self.query(X_embedded)
            K = self.key(X_embedded)    
            V = self.value(X_embedded)  

            # 3. Multi-head Attention
            Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # 2 * 8 * 10 * 64
            K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
            V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  

            # 4. Q * K^
            energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5) 

            # 5. softmax
            attention = torch.softmax(energy, dim=-1)  

            # 6. S * V 
            out = torch.matmul(attention, V)  #2 * 8 * 10 * 64

            # 7. concat 
            out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # 2 * 10 * 512

            # 8. linear after concat 
            out = self.fc_out(out)  

            return out, attention  # result, attention score