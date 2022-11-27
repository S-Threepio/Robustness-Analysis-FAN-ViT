import numpy as np
import torch


def l2_norm(adv, img):
    adv = adv.cpu().detach().numpy()
    img = img.cpu().detach().numpy()
    ret = np.sum(np.square(adv - img)) / np.sum(np.square(img))
    return ret


def linf(adv, img):
    return torch.max(torch.abs(adv.cpu() - img.cpu())).detach().numpy()
