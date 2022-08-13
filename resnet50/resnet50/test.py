from torchsummary import summary
from nets.resnet50 import resnet50
import torch
if __name__ == "__main__":
    model = resnet50(num_classes=5).train().cuda()
    summary(model,(3,299,299))
    
