#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from torchsummary import summary

from nets.resnet50 import resnet50

if __name__ == "__main__":
    model = resnet50(num_classes=10).train().cuda()
    summary(model,(3, 224, 224))
