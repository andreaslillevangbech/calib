import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms.functional import crop

def cropNarrow(im):
    return crop(im, 85, 102, 35, 224 - 2*102)

"""
width, height = im.size
top = 70
left = int(width // 2.5)
flo = crop(flo, top, margin, 80, 50)
cropnet_tsfm = 
"""

class CropNet(nn.Module):
    
    def __init(self):
        super(CropNet, self).__init__()
        self.fc1 = nn.Linear(3 * 20 * 35, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class VoxelFlow(nn.Module):

    def __init__(self):
        super(VoxelFlow, self).__init__()
        self.input_mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        self.input_std = [0.5 * 255, 0.5 * 255, 0.5 * 255]

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            6, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2_bn = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
            128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(256)

        self.bottleneck = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(256)

        # self.deconv1 = nn.Conv2d(
        #     512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.deconv1_bn = nn.BatchNorm2d(256)

        self.deconv2 = nn.Conv2d(
            256, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv2_bn = nn.BatchNorm2d(128)

        self.deconv3 = nn.Conv2d(
            128, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv3_bn = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)
        
        self.fc1 = nn.Linear(3 * 224 * 224, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 m.weight.data.normal_(0, 0.01)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

    def forward(self, x):
        input = x
        input_size = tuple(x.size()[2:4])

        x = self.conv1(x)
        x = self.conv1_bn(x)
        conv1 = self.relu(x)

        x = self.pool(conv1)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        conv2 = self.relu(x)

        x = self.pool(conv2)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        conv3 = self.relu(x)

        x = self.pool(conv3)

        x = self.bottleneck(x)
        x = self.bottleneck_bn(x)
        x = self.relu(x)

        # x = nn.functional.upsample(
        #     x, scale_factor=2, mode='bilinear', align_corners=False)

        # x = torch.cat([x, conv3], dim=1)
        # x = self.deconv1(x)
        # x = self.deconv1_bn(x)
        # x = self.relu(x)

        # x = nn.functional.upsample(
        #     x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv2], dim=1)
        x = self.deconv2(x)
        x = self.deconv2_bn(x)
        x = self.relu(x)

        # x = nn.functional.upsample(
        #     x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv1], dim=1)
        x = self.deconv3(x)
        x = self.deconv3_bn(x)
        x = self.relu(x)

        x = self.conv4(x)
        # x = nn.functional.tanh(x)

        x = torch.flatten(x, 1)
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        x = self.fc4(x)

        return x
    

if __name__=="__main__":
    model = VoxelFlow()
    print(model)
