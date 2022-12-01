import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Displays a progress bar
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
from FAN.models import fan
from timm.models import create_model
import json
import random
import foolbox as fb
from foolbox import PyTorchModel
from foolbox.attacks import EADAttack
from transferability.FGSM import testFGSM
from transferability.PGD import testPGD
from transferability.BIM import testBIM
from transferability.Mim import testMiM
import time
import timm

with open("labels.txt") as f:
    data = f.read()
# reconstructing the data as a dictionary
ground_labels = json.loads(data)

epsilon = 25 / 255
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "./pretrained_models/fan_vit_base.pth.tar"


print("Loading datasets...")
imagenet_path = r"./val"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)
imagenet_data = datasets.ImageFolder(imagenet_path, transform=transform)
val_data_loader = DataLoader(imagenet_data, batch_size=1, shuffle=True, num_workers=0)

# load model
model = create_model("fan_base_18_p16_224", img_size=224).to(device)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()


baselineModel = torch.hub.load(
    "facebookresearch/deit:main", "deit_base_patch16_224", pretrained=True
).to(device)

baselineModel.eval()

testFGSM(model, baselineModel, device, val_data_loader, epsilon)

for steps in [1, 2, 5, 10]:
    testPGD(model, baselineModel, device, val_data_loader, epsilon, steps)

for steps in [1, 2, 5, 10]:
    testBIM(model, baselineModel, device, val_data_loader, epsilon, steps)

testMiM(model, baselineModel, device, val_data_loader, epsilon, 10)


# def show_adversarial_images(examples, labels, initial_name):
#     for j in range(len(examples)):
#         plt.xticks([], [])
#         plt.yticks([], [])
#         orig, adv, ex = examples[j]
#         plt.title("{} -> {}".format(labels[str(orig)], labels[str(adv)]))
#         plt.imshow(ex[0].T)
#         name = initial_name + time.strftime("%Y%m%d-%H%M%S") + ".png"
#         plt.savefig("./adv_images/" + name)
#         plt.show()


# show_adversarial_images(ex_FGSM, ground_labels, "fsgm_")
# show_adversarial_images(ex_PGD, ground_labels, "pgd_")
# show_adversarial_images(ex_BIM, ground_labels, "bim_")
