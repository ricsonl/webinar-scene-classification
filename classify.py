from semantic_seg import ret_segm_img
import models.concater as concater
import os
import json
import argparse
import torchvision
from torchvision.io import read_image, ImageReadMode
import torch
import torch.nn as nn
import torchvision.transforms as transforms

def load_model(name):
    model_parts_path = os.path.join('models', f'{name}_parts')
    modelbest_path = os.path.join('models', f'{name}.pth.tar')

    if not os.path.isfile(modelbest_path):
        print('Concatenating model parts...')
        concater.concat_pth_tar(model_parts_path, modelbest_path)

    num_classes = 19

    model = torchvision.models.__dict__['resnet50'](pretrained=False)
    model_best = torch.load(modelbest_path, map_location=torch.device('cpu'))

    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, num_classes))
    model.load_state_dict(model_best['state_dict'])

    return model

def classify(img_path):
    model = load_model('sun-CO')
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to input image')
    args = vars(parser.parse_args())

    output = classify(args['input'])
    print(output)

if __name__ == '__main__':
    main()