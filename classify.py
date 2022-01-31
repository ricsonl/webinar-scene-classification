from semantic_seg import ret_segm_img
import models.concater as concater
import os
import json
import argparse
from torchvision.io import read_image, ImageReadMode
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

def load_model():
    modelbest_path = os.path.join('models', 'modelbest.pth.tar')
    places365_res50_path = os.path.join('models', 'places365_res50.pth.tar')

    concater.concat_pth_tar('modelbest_parts', modelbest_path)
    concater.concat_pth_tar('pre_resnet50_places365_parts', places365_res50_path)

    num_classes = 19
    num_linear = 1

    model = models.__dict__['resnet50'](pretrained=False)
    model_best = torch.load(modelbest_path, map_location=torch.device('cpu'))
    places365_res50 = torch.load(places365_res50_path, map_location=torch.device('cpu'))
    
    state_dict = places365_res50['state_dict']
    new_state_dict = OrderedDict()

    for key in state_dict.keys():
        new_state_dict[key[7:]] = state_dict[key]

    model.fc = nn.Linear(2048, 365)
    model.load_state_dict(new_state_dict)
    layers = []
    curr = model.fc.in_features
    dif = round((curr - num_classes + 1) / num_linear)
    for i in range(num_linear-1):
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(curr, curr - dif))
        layers.append(nn.LeakyReLU())
        curr = curr - dif
    layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(curr, num_classes))
    model.fc = nn.Sequential(*layers)

    model.load_state_dict(model_best['state_dict'])

    return model

def classify(model, img):
    model.eval()
    softmax = nn.Softmax(dim=1)

    with open('classes.json') as json_file:
        classes_dict = json.load(json_file)

    output = model(img)
    output = softmax(output)
    max, preds = torch.max(output, 1)
    print(max, preds)

    if max[0] >= 0.7:
        return classes_dict[preds[0]]

    return 'uncertain'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to input image')
    args = vars(parser.parse_args())
    img = read_image(args['input'], mode=ImageReadMode.RGB)

    img_segm = ret_segm_img(img)

    model = load_model()

    print(classify(model, img_segm.unsqueeze(0)))


if __name__ == '__main__':
    main()