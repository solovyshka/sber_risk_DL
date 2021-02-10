import torch
import numpy as np
import cv2

from torchvision import models, transforms
from PIL import Image

from week16.CAM_vis import CAM_localization
from week16.imagenet_class_map import imagenet_classmap


def get_classmap(class_idx, last_convs, weight_softmax):
    h = w = 7
    nc = 2048
    reshape_convs = torch.squeeze(last_convs[0], 0).reshape((nc, h * w)).detach().numpy()
    classmap = weight_softmax[class_idx].dot(reshape_convs)
    classmap = np.reshape(classmap, [h, w])

    return classmap

def model_forward(batch_t):
    resnet = models.resnet50(pretrained=True)
    params = list(resnet.parameters())
    weights_softmax = np.squeeze(params[-2].data.numpy())

    last_convs = []
    def hook(module, input, output):
        last_convs.append(output)

    res50_model = models.resnet50(pretrained=True)
    res50_model.layer4[2].conv3.register_forward_hook(hook)

    out = res50_model(batch_t)
    class_idx = np.argmax(out[0].detach().numpy())
    print(class_idx, imagenet_classmap[class_idx])

    return class_idx, last_convs, weights_softmax

def show_CAM(path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )]
    )

    img = Image.open(path)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    class_idx, last_convs, weights_softmax = model_forward(batch_t)

    classmap = get_classmap(class_idx, last_convs, weights_softmax)

    img2 = cv2.imread(path, cv2.IMREAD_COLOR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    cam = CAM_localization()
    cam.get_localization_map(img2, classmap)

    return


if __name__ == '__main__':
    show_CAM(path ='./data/dog.jpg')


