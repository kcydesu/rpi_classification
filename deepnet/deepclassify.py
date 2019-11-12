from torchvision import models
import torch
from PIL import Image
from torchvision import transforms


class SexyNet:
    def __init__(self, img='dog.jpg'):
        self._alexnet = models.alexnet(pretrained=True)
        self.img_file = img
        self.classProbs = []
        self.classNames = []
        self._transform = transforms.Compose([  # [1]
            transforms.Resize(256),  # [2]
            transforms.CenterCrop(224),  # [3]
            transforms.ToTensor(),  # [4]
            transforms.Normalize(  # [5]
                mean=[0.485, 0.456, 0.406],  # [6]
                std=[0.229, 0.224, 0.225]  # [7]
            )])

    def eval(self):
        self.classProbs = []
        self.classNames = []

        img = Image.open(self.img_file)
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        self._alexnet.eval()
        out = self._alexnet(batch_t)

        with open('imagenet_classes.txt') as f:
            labels = [line.strip() for line in f.readlines()]

        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        _, indices = torch.sort(out, descending=True)
        for idx in indices[0][:5]:
            self.classNames.append(labels[idx])
            self.classProbs.append(percentage[idx])

    def getclass(self):
        return self.classNames[0], self.classProbs[0]

    def getname(self):
        return self.classNames[0]

    def getprob(self):
        return self.classProbs[0]

    def topprob(self, n=5):
        for i in range(5):
            print(self.classNames[i], self.classProbs[i])