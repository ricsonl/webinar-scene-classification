from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
from torchvision.transforms.functional import convert_image_dtype
from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms.functional as F
import torch
import argparse

def ret_segm_img(img):
    model = fcn_resnet50(pretrained=True)
    model = model.eval()

    img = convert_image_dtype(img, dtype=torch.float)
    normalized = F.normalize(img.unsqueeze(0), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    output = model(normalized)['out']

    sem_classes = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    normalized_mask = torch.nn.functional.softmax(output, dim=1)
    class_dim = 1
    boolean_person_mask = (normalized_mask.argmax(class_dim) == sem_class_to_idx['person'])

    output = img * ~boolean_person_mask

    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to input image')
    args = vars(parser.parse_args())

    img = read_image(args['input'], mode=ImageReadMode.RGB)

    save_image(ret_segm_img(img), f'segm_output/{args["input"].split("/")[-1]}')

if __name__ == '__main__':
    main()