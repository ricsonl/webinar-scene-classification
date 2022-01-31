import model.concater as concater
import os
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

def main():
    concater.concat_pth_tar()

    # num_classes = 19
    # num_linear = 1

    # model_best_path = os.path.join('model', 'modelbest.pth.tar.gz')

    # model = models.__dict__['resnet50'](pretrained=False)
    # model_best = torch.load(model_best_path, map_location=torch.device('cpu'))
    # model.load_state_dict(model_best['state_dict'])

    # print(model)

if __name__ == '__main__':
    main()