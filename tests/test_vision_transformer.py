import pytest
import torch
import torch.nn as nn
from einops import rearrange
from src.vit import Attention, PatchEmbedding, RotaryPositionEmbedding, VisionTransformer, MLP, Block
TEST_CONFIG = {
    'img_size': 224,
    'patch_size': 16,
    'in_c': 3,
    'embed_dim': 128,  # 减小尺寸以适应 macOS 内存
    'depth': 4,        # 减少层数以加快测试速度
    'num_heads': 8,
    'num_class': 10    # 减少类别数以加快测试
}
# Fixtures
@pytest.fixture
def test_image():
    return torch.randn(1, 3, 224, 224)  # 减小batch size以节省内存

@pytest.fixture
def small_test_image():
    return torch.randn(1, 3, 64, 64)

@pytest.fixture
def default_model():
    return VisionTransformer(**TEST_CONFIG)

@pytest.fixture
def small_model():
    config = TEST_CONFIG.copy()
    config.update({
        'img_size': 64,
        'embed_dim': 64,
        'depth': 2,
        'num_heads': 4
    })
    return VisionTransformer(**config)

# 测试函数
def test_patch_embedding():
    embed = PatchEmbedding(224, 16, 3, 128)
    x = torch.randn(1, 3, 224, 224)
    output = embed(x)
    assert output.shape == (1, 196, 128)  # (224/16)^2 = 196 patches

def test_rotary_position_embedding():
    rope = RotaryPositionEmbedding(dim=64)
    x = torch.randn(1, 4, 10, 64)  # [batch, heads, seq_len, dim]
    output = rope(x)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()

def test_attention_layer():
    attn = Attention(embed_dim=128, num_heads=8)
    x = torch.randn(1, 10, 128)
    output = attn(x)
    assert output.shape == x.shape

def test_mlp_layer():
    mlp = MLP(128, 128*4, 128, 0)
    x = torch.randn(1, 10, 128)
    output = mlp(x)
    assert output.shape == x.shape

def test_transformer_block():
    block = Block(dim=128, num_heads=8)
    x = torch.randn(1, 10, 128)
    output = block(x)
    assert output.shape == x.shape

@pytest.mark.parametrize("batch_size", [1, 2])  # 限制batch size
def test_vit_forward_pass(default_model, batch_size):
    x = torch.randn(batch_size, 3, 224, 224)
    output = default_model(x)
    assert output.shape == (batch_size, TEST_CONFIG['num_class'])
    assert not torch.isnan(output).any()

def test_vit_output_range(small_model, small_test_image):
    output = small_model(small_test_image)
    assert output.abs().max().item() < 100  # 检查输出值范围

def test_vit_gradient_flow(small_model, small_test_image):
    x = small_test_image.clone().requires_grad_(True)
    output = small_model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

def test_vit_training_mode(default_model, test_image):
    default_model.train()
    output = default_model(test_image)
    assert output.requires_grad
    assert default_model.training

def test_vit_eval_mode(default_model, test_image):
    default_model.eval()
    with torch.no_grad():
        output = default_model(test_image)
    assert not output.requires_grad
    assert not default_model.training

@pytest.mark.parametrize("patch_size", [8, 16])  # 减少测试参数以加快速度
def test_vit_with_different_patch_sizes(patch_size):
    config = TEST_CONFIG.copy()
    config['patch_size'] = patch_size
    model = VisionTransformer(**config)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    assert output.shape == (1, config['num_class'])

def test_vit_with_odd_dimensions():
    with pytest.raises(AssertionError):
        RotaryPositionEmbedding(dim=63)  # 测试奇数维度

