import os
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.utils import GradCAM, show_cam_on_image,letterbox_image
from nets.resnet50 import resnet50

def detect_image(img_path):  
    model = resnet50(num_classes=3)
    # print(model)
    weights_path = r"logs\Epoch27-Total_Loss0.0011-Val_Loss0.0004.pth"
    model.load_state_dict(torch.load(weights_path))
    
    target_layers = [model.layer4] # 在这里可以查看四层分别是 layer 1234
    data_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])    
    img = letterbox_image(img_path, [299,299])
    img = np.array(img, dtype=np.uint8)
    
    # img = center_crop_img(img, 299)

    # [N, C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    # target_category = 0
    grayscale_cam = cam(input_tensor=input_tensor, target_category=None)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.show()
      
if __name__ == '__main__':
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            class_name = detect_image(image)
