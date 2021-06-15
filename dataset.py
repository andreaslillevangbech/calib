import torch
import numpy as np
import os
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms.transforms import Normalize

class VideoData(Dataset):
    def __init__(self, src_dir, vid, transforms=None):
        vid = str(vid)
        img_paths = os.listdir(os.path.join(src_dir, vid))
        self.img_paths = [os.path.join(src_dir, vid, img_path) \
                                    for img_path in img_paths]
        self.y = np.loadtxt(os.path.join(src_dir , vid + '.txt' ))
        self.transforms = transforms

    def __getitem__(self, idx):
        y = self.y[idx]
        y2 = self.y[idx+1]
        y = torch.Tensor((y+y2)/2)

        img = Image.open(self.img_paths[idx]) # .convert('L')
        img2 = Image.open(self.img_paths[idx+1]) # .convert('L')

        if self.transforms:
            img = self.transforms(img)
            img2 = self.transforms(img2)
            img = torch.cat((img, img2), 0)

        return img, y
        
    def __len__(self):
        return len(self.y) - 1


class FlowData(Dataset):
    def __init__(self, src_dir, vid, transforms=None):
        vid = str(vid)
        folder = 'flows' + vid
        img_paths = os.listdir(os.path.join(src_dir, folder))
        self.img_paths = [os.path.join(src_dir, folder, img_path) \
                                    for img_path in img_paths]
        self.y = np.loadtxt(os.path.join(src_dir , vid + '.txt' ))
        self.transforms = transforms

    def __getitem__(self, idx):
        y = self.y[idx]
        y2 = self.y[idx+1]
        y = torch.Tensor((y+y2)/2)
        img = Image.open(self.img_paths[idx])

        if self.transforms:
            img = self.transforms(img)

        return img, y
        
    def __len__(self):
        return len(self.y) - 1
        
if __name__=="__main__":
    trans = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.2]),]) # Greyscale
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    dat = FlowData('labeled', '0', trans)
    print(dat[0][0].shape)



