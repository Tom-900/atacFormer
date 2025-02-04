import torch
from torch import nn
from flash_attn.flash_attn_interface import flash_attn_func

# Input shapes
batch_size = 1
L = 10  # Sequence length for q
d_qk = 32  # Query/key dimension
d_kv = 512  # Key/value dimension

# Input tensors
q = torch.randn(batch_size, L, d_qk).cuda()  # Shape (1, L, 32)
k = torch.randn(batch_size, 1, d_kv).cuda()  # Shape (1, 1, 512)
v = torch.randn(batch_size, 1, d_kv).cuda()  # Shape (1, 1, 512)

# Project k and v to match the query dimension (32)
k_proj = nn.Linear(d_kv, d_qk).cuda()
v_proj = nn.Linear(d_kv, d_qk).cuda()

k = k_proj(k)  # Shape (1, 1, 32)
v = v_proj(v)  # Shape (1, 1, 32)

qkv = torch.cat([q, k, v], dim=1)  # Shape (1, L+1+1, 32)
cu_seqlens = torch.tensor([0, L + 1 + 1], dtype=torch.int32).cuda()

# Scale factor for attention
scale = 1.0 / (d_qk ** 0.5)

# Compute attention
attn_output, _ = flash_attn_func(qkv, cu_seqlens, L + 1 + 1, 0.0, causal=False)

print("Attention Output Shape:", attn_output.shape)  # Expected: (1, L, 32)