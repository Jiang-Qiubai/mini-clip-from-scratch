import pytest
import torch
from torch import nn

# 从你的模块导入
from src.vit import Attention

import torch
from einops import rearrange
from torch import nn
import pytest

from src.vit import Attention  # 替换为你的实际模块路径

@pytest.fixture
def attention_config():
    return {
        'num_heads': 12,
        'embed_dim': 768
    }

@pytest.fixture
def input_tokens():
    batch_size = 4
    num_patches = 196
    embed_dim = 768
    return torch.randn(batch_size, num_patches, embed_dim)

def test_attention_output_shape(attention_config, input_tokens):
    attention = Attention(**attention_config)
    output = attention(input_tokens)
    
    assert output.shape == input_tokens.shape, \
        f"输出形状应与输入形状相同 {input_tokens.shape}, 但得到 {output.shape}"

def test_attention_qkv_linear(attention_config):
    attention = Attention(**attention_config)
    
    assert isinstance(attention.qkv_linear, nn.Linear), "qkv_linear 应该是 Linear 层"
    assert attention.qkv_linear.out_features == 3 * attention_config['embed_dim'], \
        "qkv_linear 输出特征应为 3 * embed_dim"

def test_attention_proj_layer(attention_config):
    attention = Attention(**attention_config)
    
    assert isinstance(attention.proj, nn.Linear), "proj 应该是 Linear 层"
    assert attention.proj.in_features == attention_config['embed_dim'], \
        "proj 输入特征应与 embed_dim 匹配"
    assert attention.proj.out_features == attention_config['embed_dim'], \
        "proj 输出特征应与 embed_dim 匹配"

def test_attention_head_dim(attention_config):
    attention = Attention(**attention_config)
    
    expected_head_dim = attention_config['embed_dim'] // attention_config['num_heads']
    assert attention.head_dim == expected_head_dim, \
        f"head_dim 应为 {expected_head_dim}, 但得到 {attention.head_dim}"

def test_attention_scale(attention_config):
    attention = Attention(**attention_config)
    
    expected_scale = (attention_config['embed_dim'] // attention_config['num_heads']) ** -0.5
    assert abs(attention.scale - expected_scale) < 1e-6, \
        f"scale 应为 {expected_scale}, 但得到 {attention.scale}"