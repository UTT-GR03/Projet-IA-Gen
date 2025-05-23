# helpers/bsrgan_inference.py
import torch
import os
from network_rrdbnet import RRDBNet as net
from utils import utils_image as util

def load_bsrgan_model(model_path='BSRGAN.pth', scale=4, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=scale)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    return model

def run_bsrgan_inference(pil_image, model, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = np.array(pil_image.convert('RGB'))  # PIL to ndarray
    img_L = util.uint2tensor4(img)            # to torch tensor
    img_L = img_L.to(device)
    with torch.no_grad():
        img_E = model(img_L)
    img_E = util.tensor2uint(img_E)           # to numpy uint8
    return img_E