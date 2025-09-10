import torch
from einops import rearrange
from torch import nn
import pytest

from src.vit import PatchEmbedding  # 替换为你的实际模块路径

@pytest.fixture
def patch_embedding_config():
    return {
        'img_size': 224,
        'patch_size': 16, 
        'in_channels': 3,
        'embed_dim': 768
    }

@pytest.fixture
def input_image():
    batch_size = 4
    img_size = 224
    return torch.randn(batch_size, 3, img_size, img_size)

def test_patch_embedding_output_shape(patch_embedding_config, input_image):
    patch_embed = PatchEmbedding(**patch_embedding_config)
    output = patch_embed(input_image)
    
    batch_size = input_image.shape[0]
    patch_size = patch_embedding_config['patch_size']
    img_size = input_image.shape[2]  # 假设是正方形图像
    num_patches = (img_size // patch_size) ** 2
    embed_dim = patch_embedding_config['embed_dim']
    
    assert output.shape == (batch_size, num_patches, embed_dim), \
        f"输出形状应为 {(batch_size, num_patches, embed_dim)}, 但得到 {output.shape}"

def test_patch_embedding_proj_layer(patch_embedding_config):
    patch_embed = PatchEmbedding(**patch_embedding_config)
    
    assert isinstance(patch_embed.proj, nn.Conv2d), "proj 应该是 Conv2d 层"
    assert patch_embed.proj.kernel_size == (patch_embedding_config['patch_size'],) * 2, \
        "卷积核大小应与 patch_size 匹配"
    assert patch_embed.proj.stride == (patch_embedding_config['patch_size'],) * 2, \
        "步长应与 patch_size 匹配"