import json

import torch
import torchvision
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

from layers import Identity

image_index = 0

with open("./imagenet_class_index.json", "r") as f:
  index2class = json.load(f)

resnet50 = torchvision.models.resnet50(pretrained=True).eval()
image_path = "./images/original_{}.jpg".format(image_index)
image_size = (224, 224)
image = Image.open(image_path).convert("RGB")
transform = torchvision.transforms.Compose([
  torchvision.transforms.Resize(image_size),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
tranformed_image = transform(image).unsqueeze(0)
class_index = torch.argmax(resnet50(tranformed_image), dim=1)
weight = resnet50.fc.weight[class_index].unsqueeze(2).unsqueeze(3).repeat(1, 1, 7, 7)

resnet50.fc = Identity()
resnet50.avgpool = Identity()
average_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
feature_map = resnet50(tranformed_image).view(1, 2048, 7, 7)
pooled_weight = average_pool(feature_map)
cam = torch.sum(feature_map * weight, dim=1).squeeze().detach().numpy()
cam = cv2.resize(cam, image_size)

plt.figure()
plt.imshow(np.asarray(image.resize(image_size)))
plt.imshow(cam, cmap="gray", alpha=0.75)
plt.set_cmap("hot")
plt.xticks([])
plt.yticks([])
plt.title("CAM {}".format(index2class[str(class_index.item())][1]))
plt.tight_layout()
plt.savefig("./cams/cam_{}.png".format(image_index))