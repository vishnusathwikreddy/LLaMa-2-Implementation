import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    
    max_seq_len: int = 2048
    max_batch_size: int = 32
    
    device: str = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len:int, device: str):
    #(head_dim/2)
    #theta_i = 10000^(-2(i-1)/dim)
    numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0/(10000.0 ** (numerator/head_dim))
    m = torch.arange(0, seq_len)
    freqs = torch.outer(m, theta).float()  #(seq_len, head_dim/2)
    #c = e^(i*m*theta)
    freqs_complex = torch.polar(torch.ones_like(freqs),freqs)
    return freqs_complex.to(device)
    
def apply_rotatory_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2)) #(batch_size, seq_len, H, head_dim) --> (batch_size, seq_len, H, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(1)  #(seq_len, head_dim/2) --> (1, seq_len, 1, head_dim/2)
    x_rotated = x_complex * freqs_complex #(batch_size, seq_len, H, head_dim/2) * (1, seq_len, 1, head_dim/2) = (batch_size, seq_len, H, head_dim/2)
    x_out = torch.view_as_real(x_rotated) #(batch_size, seq_len, H, head_dim/2) --> (batch_size, seq_len, H, head_dim/2, 2)
    x_out = x_out.reshape(*x.shape) #(batch_size, seq_len, H, head_dim/2, 2) --> (batch_size, seq_len, H, head_dim)
    return x_out

def repeat_kv(x: torch.Tensor, n_rep:int):
    batch_size, seq_len, n_kv_heads, head_dim  = x.shape
    
    if n_rep==1:
        return x
    
    return x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len, n_kv_heads*n_rep, head_dim)



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms #(batch_size, seq_len, dim) * (batch_size, seq_len, 1) --> (batch_size, seq_len, dim)
    
    def forward(self, x: torch.Tensor):
        #(dim) * (batch_size, seq_len, dim) --> (batch_size, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)
    

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = int(8*args.dim/3)
        
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim* args.ffn_dim_multiplier)
        hidden_dim = args.multiple_of*((hidden_dim + args.multiple_of -1)//args.multiple_of)
        
        self.w1 = nn.Linear(args.dim , hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self,x: torch.Tensor):
            
        #swiglu(x, W, V, W2) = (swish(xW) * xV) * W2
        #*W2 is a nn
            
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x
            
            

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()   
        self.dim = args.dim
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads # in grouped query attention kv_heads may != q_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads # indicates the no of times kv heads has to be repeated to match the head queries
        self.head_dim = self.dim // args.n_heads
        
        self.wq = nn.Linear(self.dim, args.n_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim,  self.n_kv_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads*self.head_dim, bias = False)
        self.wo  = nn.Linear(args.n_heads*self.head_dim, args.dim, bias=False)
        
        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim).to(args.device)
        self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim).to(args.device)
        
    def forward(self, x: torch.Tensor, start_pos:int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        xq = apply_rotatory_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotatory_embeddings(xk, freqs_complex, device=x.device)
        
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv
        
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]
        
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        
        xq = xq.float()
        keys = keys.float()
        values = values.float()
        
        attention_scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        out = torch.matmul(attention_scores, values)
        
        out = out.to(x.dtype)
        output = out.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        
        return self.wo(output)
    
        
class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos:int, freqs_complex: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
       
        
class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args=args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim//self.args.n_heads, self.args.max_seq_len*2, self.args.device)
    
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)
        output = self.output(h).float()
        return output

