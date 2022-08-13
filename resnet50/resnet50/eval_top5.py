import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from classification import Classification, _preprocess_input
from utils.utils import letterbox_image
from nets.confusion import ConfusionMatrix


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
classes_path = 'cls_classes.txt' 
class_names = get_classes(classes_path)

class top5_Classification(Classification):
    def detect_image(self, image):        
        crop_img = letterbox_image(image, [self.input_shape[0],self.input_shape[1]])
        photo = np.array(crop_img,dtype = np.float32)

        # 图片预处理，归一化
        photo = np.reshape(_preprocess_input(photo),[1,self.input_shape[0],self.input_shape[1],self.input_shape[2]])
        photo = np.transpose(photo,(0,3,1,2))

        with torch.no_grad():
            photo = Variable(torch.from_numpy(photo).type(torch.FloatTensor))
            if self.cuda:
                photo = photo.cuda()
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        arg_pred = np.argsort(preds)[::-1]
        arg_pred_top5 = arg_pred[:5]
        return arg_pred_top5

def evaluteTop5(classfication, lines):
    label = [label for label in class_names]
    confusion = ConfusionMatrix(num_classes=40, labels=label)
    correct = 0
    total = len(lines)
    Pre = []
    Tru = []
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])
        Tru.append(y)
        pred = classfication.detect_image(x)
        Pre.append(pred)
        
        correct += y in pred
        if index % 100 == 0:
            print("[%d/%d]"%(index,total))
    confusion.update(Tru, Pre)
    confusion.plot()
    confusion.summary()      
    return correct / total

classfication = top5_Classification()
with open(r"./cls_test.txt","r") as f:
    lines = f.readlines()
top5 = evaluteTop5(classfication, lines)
print("top-5 accuracy = %.2f%%" % (top5*100))

