import torch
import torch.nn as nn
from einops import rearrange, einsum

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_patches = (img_size//patch_size) ** 2 
    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x

class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码 (Rotary Position Embedding)"""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        assert dim % 2 == 0, "Dimension must be even for rotary position embedding"
        
        # 预计算频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算缓存
        self._compute_cos_sin_cache(max_seq_len)
    
    def _compute_cos_sin_cache(self, seq_len):
        """预计算cos和sin缓存"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        
        # 添加必要的维度以便广播
        self.register_buffer('cos_cached', emb.cos()[None, :, None, :])  # [1, seq_len, 1, dim]
        self.register_buffer('sin_cached', emb.sin()[None, :, None, :])  # [1, seq_len, 1, dim]
    
    def forward(self, x, seq_dim=1):
        """应用旋转位置编码
        Args:
            x: 输入张量 [batch, seq_len, heads, dim] 或 [batch, heads, seq_len, dim]
            seq_dim: 序列长度的维度
        """
        seq_len = x.shape[seq_dim]
        
        # 如果序列长度超过缓存大小，重新计算
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len * 2
            self._compute_cos_sin_cache(self.max_seq_len)
        
        # 获取对应序列长度的cos和sin
        cos = self.cos_cached[:, :seq_len, :, :]  # [1, seq_len, 1, dim]
        sin = self.sin_cached[:, :seq_len, :, :]  # [1, seq_len, 1, dim]
        
        # 调整维度顺序以匹配输入
        if seq_dim == 2:  # 如果seq_len在dim=2
            cos = cos.permute(0, 2, 1, 3)  # [1, 1, seq_len, dim]
            sin = sin.permute(0, 2, 1, 3)  # [1, 1, seq_len, dim]
        
        # 将x的后半部分取负
        x1, x2 = x.chunk(2, dim=-1)
        x_rot = torch.cat((-x2, x1), dim=-1)
        
        # 应用旋转
        return (x * cos) + (x_rot * sin)
    
class Attention(nn.Module):
    def __init__(self, num_heads, embed_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.qkv_linear = nn.Linear(embed_dim, 3*embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.head_dim = self.embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rotary_pos_embed = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x):
        qkv = self.qkv_linear(x)
        qkv = rearrange(qkv, 'b n (three num_heads head_dim) ->\
        three b num_heads n head_dim', three=3, num_heads=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rotary_pos_embed(q)
        k = self.rotary_pos_embed(k)
        attn = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale
        attn = attn.softmax(dim=-1)

        x = einsum(attn, v, "b h i j, b h j d -> b h i d")
        x = rearrange(x, 'b h i d -> b i (h d)')

        x = self.proj(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.act_fun = nn.GELU()
        self.layer2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fun(x)
        x = self.drop(x)
        x = self.layer2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(num_heads, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, 4*dim, dim, 0)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=27, num_class=8, patch_size=3, in_c=1, embed_dim=16, depth=2,
                 num_heads=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed = PatchEmbedding(img_size, patch_size, in_c, embed_dim)
        self.rotary_pos_embed = RotaryPositionEmbedding(embed_dim // num_heads)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.block = nn.Sequential(*[
            Block(embed_dim, num_heads) for i in range(depth)
        ])
        num_patches = self.embed.num_patches
        self.num_class = num_class
        self.head = nn.Linear(embed_dim, num_class)
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+1,embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed,std=0.02)
        
        nn.init.trunc_normal_(self.cls_token,std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        x = self.embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.block(x)
        x = self.norm(x)
        x = self.head(x[:,0])
        return x
    
def _init_vit_weights(m):
    # 判断模块m是否是nn.linear
    if isinstance(m,nn.Linear):
        nn.init.trunc_normal_(m.weight,std=.01)
        if m.bias is not None: # 如果线性层存在偏置项
            nn.init.zeros_(m.bias)

    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode="fan_out") # 对卷积层的权重做一个初始化 适用于卷积
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m,nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight) # 对层归一化的权重初始化为1