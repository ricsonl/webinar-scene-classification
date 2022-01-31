from semantic_seg import ret_segm_img
import models.concater as concater
import os
import json
import torchvision
from torchvision.io import read_image, ImageReadMode
import torch
import torch.nn as nn
import torchvision.transforms as transforms

def load_model():
    modelbest_path = os.path.join('models', 'modelbest.pth.tar')
    concater.concat_pth_tar('modelbest_parts', modelbest_path)

    num_classes = 19

    model = torchvision.models.__dict__['resnet50'](pretrained=False)
    model_best = torch.load(modelbest_path, map_location=torch.device('cpu'))

    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, num_classes))
    model.load_state_dict(model_best['state_dict'])

    return model

def classify(img_path):
    model = load_model()
    img_segm = ret_segm_img( read_image(img_path, mode=ImageReadMode.RGB) )

    transform = transforms.Compose([
        transforms.Normalize(   mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])

    img_inp = transform(img_segm).unsqueeze(0)

    model.eval()
    softmax = nn.Softmax(dim=1)

    output = model(img_inp)
    output = softmax(output)
    max, preds = torch.max(output, 1)
    
    with open('classes.json') as json_file:
        classes_dict = json.load(json_file)

    return (classes_dict[str(preds[0].item())], max[0].item())