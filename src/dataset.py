from torch.utils.data import Dataset
from torchvision.transforms.v2 import PILToTensor,Compose
import torchvision

# 手写数字
class MNIST(Dataset):
    def __init__(self,is_train=True):
        super().__init__()
        self.ds=torchvision.datasets.MNIST('./mnist/',train=is_train,download=True)
        self.img_convert=Compose([
            PILToTensor(),
        ])
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,index):
        img,label=self.ds[index]
        return self.img_convert(img)/255.0,label