import numpy as np
import torch
import pickle
from multiprocessing import process
import sys
from math import pi
from PIL import Image
import cv2 as cv
from codes import GAN_class
from torch import nn


def get_in(img):
    tmp = np.swapaxes(img, 2, 0)
    return np.swapaxes(tmp, 1, 2)

def get_out(img):
    tmp = np.swapaxes(img, 2, 0)
    return np.swapaxes(tmp, 1, 0)

def add_noise(inp, im_sz=(192, 192)):
    hr_noise = torch.rand(inp.size()[0], 1, im_sz[0], im_sz[1])
    return torch.cat((inp, hr_noise), 1) 


def save_model(model, path):
    with open(path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

def load_model(path):
    with open(path, 'rb') as input:
        model = pickle.load(input)
    assert model is not None
    return model

def bicubic6x(img):
    res = img#.filter(ImageFilter.GaussianBlur(radius = 2/pi)) 
    res_w, res_h = res.size
    for i in range(5):
        width, height = res.size
        res = res.resize((int(width * 0.8909), int(height * 0.8909)), Image.BICUBIC)
        
    return np.array(res.resize((res_w // 2, res_h // 2),Image.BICUBIC)) / 255
    

def read_img(path):
    img = cv.imread(path)
    img = torch.tensor(get_in(img)).float()
    img = img.reshape(1, img.size(0), img.size(1), img.size(2)) 
    return add_noise(img, [img.size(2), img.size(3)]) 


def create_generator(path, device, ngpu=1, num_channels=3):
    # Create the Generator
    generator = GAN_class.Generator(num_channels)
    generator.load_state_dict(torch.load(path))
    generator.eval()
    generator.to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        generator = nn.DataParallel(generator, list(range(ngpu)))
    _ = generator.apply(GAN_class.weights_init)
    return generator