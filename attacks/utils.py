import numpy as np
import torch


def l2_norm(adv, img):
    adv = adv.cpu().detach().numpy()
    img = img.cpu().detach().numpy()
    ret = np.sum(np.square(adv - img)) / np.sum(np.square(img))

    if ret:
        return ret
    else:
        return 0


def linf(adv, img):
    ret = torch.max(torch.abs(adv.cpu() - img.cpu())).detach().numpy()
    if ret:
        return ret
    else:
        return 0
