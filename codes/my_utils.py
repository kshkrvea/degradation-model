import numpy as np
import torch
import pickle

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