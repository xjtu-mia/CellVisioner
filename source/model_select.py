import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import numpy as np

# 预处理
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     ])  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def model_select_main(user_name):
    # create model
    # model = vgg(num_classes=4)
    dataset_path =os.path.join(f'./datasets',user_name,r'pbda/train/bright')
    result_path=os.path.join('./results',user_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    model_name = 'vgg16'
    model = _vgg(model_name=model_name, num_classes_s=4, pretrained=False, init_weights=True)
    # load model weights
    model_weight_path = "./model_weight/vgg16Net.pth"
    model.load_state_dict(torch.load(model_weight_path))
    
    # read class_indict
    try:
        json_file = open('./model_weight/class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)
    
    # load image
    data_dir = os.listdir(dataset_path) if len(os.listdir(dataset_path))<50 else os.listdir(dataset_path)[:50]
    with open(fr'{result_path}/result.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
     
        class_dict = {}
        for img_name in data_dir:
            image_path = os.path.join(dataset_path, img_name)
            img = cv2.imread(image_path)
            m = np.mean(img)
            std = np.std(img)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = data_transform(img)
            img = transforms.Normalize((m, m, m), (std, std, std))(img)
            img = torch.unsqueeze(img, dim=0)
            model.eval()
            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img))
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
            class_name, probability = class_indict[str(predict_cla)], predict[predict_cla].item()
            # print(class_name, probability)
            if class_name not in class_dict.keys():
                class_dict[class_name] = 0
            class_dict[str(class_name)] += probability

        f.writelines('------------------- end -------------------' + '\n')
        print(str(class_dict))
        f.writelines(str(class_dict) + '\n')
        f.writelines('------------------- end -------------------' + '\n')
        class_dict=sorted(class_dict.items(),key=lambda x:x[1],reverse=True)
    return class_dict[0][0]
        
    






class VGG(nn.Module):
    
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(model_name, num_classes_s, pretrained, **kwargs):
    model = VGG(make_layers(cfgs[model_name], batch_norm=False), **kwargs)
    if pretrained:
        model.features.load_state_dict(torch.load(r'./Model_select/vgg16-397923af.pth'), strict=False)
    num_fc = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_fc, num_classes_s)
    
    return model