import torch
import matplotlib.pyplot as plt
from clip import CLIP
from dataset import MNIST
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# 初始化
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLIP().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 加载数据
dataset = MNIST()
img, true_label = dataset[0]  # 取第一张图片测试

# 1. 图片分类
print(f"真实标签: {true_label}")
plt.imshow(img.permute(1,2,0))
plt.title(f"待分类图片 (真实标签: {true_label})")
plt.show()

# 预测0-9哪个数字
logits = model(img.unsqueeze(0).to(device), torch.arange(10).to(device))
pred_label = logits.argmax(-1).item()
print(f"预测结果: {pred_label}")