import torch 
import torch.nn as nn
import torch.nn.functional as F
import math


class TokenRouter(nn.Module):
    """Routes top-k tokens to deep processing path"""
    def __init__(self, dim, capacity=0.25):
        super().__init__()
        self.gate = nn.Linear(dim, 1)
        self.capacity = capacity
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        scores = self.gate(x).squeeze(-1)
        
        k = max(1, int(seq_len * self.capacity))
        topk_values, topk_indices = torch.topk(scores, k, dim=-1)
        
        route_mask = torch.zeros_like(scores, dtype=torch.bool)
        route_mask.scatter_(1, topk_indices, True)
        
        return route_mask, scores


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with grouped key-value heads (GQA)"""
    def __init__(self, dim, n_heads=4, n_kv_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = F.softmax(att, dim=-1)
        out = att @ v
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SimplifiedSSM(nn.Module):
    """Lightweight State Space Model approximation"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.A = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.B = nn.Linear(dim, dim, bias=False)
        self.C = nn.Linear(dim, dim, bias=False)
        self.D = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        outputs = []
        
        for t in range(L):
            h = torch.tanh(h @ self.A.T + self.B(x[:, t]))
            y = self.C(h) + self.D * x[:, t]
            outputs.append(y)
            
        return torch.stack(outputs, dim=1)


class HybridLayer(nn.Module):
    """Layer with dynamic routing and hybrid attention-SSM"""
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.router = TokenRouter(dim, capacity=0.25)
        
        self.attention = GroupedQueryAttention(dim, n_heads=n_heads, n_kv_heads=1)
        self.ssm = SimplifiedSSM(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.skip_gate = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        route_mask, scores = self.router(x)
        residual = x
        
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm)
        ssm_out = self.ssm(x_norm)
        deep_out = attn_out + ssm_out
        
        route_mask_expanded = route_mask.unsqueeze(-1).float()
        x = residual + deep_out * route_mask_expanded + residual * self.skip_gate * (1 - route_mask_expanded)
        
        x = x + self.ffn(self.norm2(x))
        
        return x, scores


class HyperFocusLM(nn.Module):
    """Complete HyperFocus Language Model"""
    def __init__(self, vocab_size=10000, dim=256, n_layers=4, n_heads=4, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([HybridLayer(dim, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        
    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embedding(input_ids) + self.pos_embed[:, :T, :]
        
        routing_scores = []
        for layer in self.layers:
            x, scores = layer(x)
            routing_scores.append(scores)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, routing_scores
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


class BaselineTransformer(nn.Module):
    """Standard Transformer for comparison"""
    def __init__(self, vocab_size=10000, dim=256, n_layers=4, n_heads=4, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim*4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        
    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embedding(input_ids) + self.pos_embed[:, :T, :]
        
        causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.norm(x)
        
        return self.lm_head(x), None
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())