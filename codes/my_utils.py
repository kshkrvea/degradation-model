import numpy as np
import torch
import pickle
from PIL import Image
import cv2 as cv
from codes import GAN_class
from torch import nn
from codes.GAN_class import DownsampleGAN

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
    generator = GAN_class.Generator()
    generator.load_state_dict(torch.load(path))
    generator.eval()
    generator.to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        generator = nn.DataParallel(generator, list(range(ngpu)))
    generator.apply(GAN_class.weights_init)
    return generator



def hr2lr(netG, device, inp):
    inp = torch.tensor(get_in(inp)).float()
    inp = inp.resize(1, inp.size(0), inp.size(1), inp.size(2))
    inp = add_noise(inp, [inp.size(2), inp.size(3)])
    res = netG(inp.to(device))
    res = get_out(res.detach().cpu().numpy()[0])
    return res

def concat_vh(list_2d):
    return cv.hconcat([cv.vconcat(list_h) for list_h in list_2d])

def frame_processing(netG, device, path, block_size=512):
    ''' 
    > the frame size often exceeds the amount of memory in the GPU, so we have to divide the image into clippings (size of each cropped block <= block_size) 
    and process each cropping individually
    > this function contains all stages of degraiding (applying [netG] generator) image stored at [path]
    > returnes degraded full frame
    '''
    board_HR = cv.imread(path)[... , ::-1] / 255
    LR_dim = np.array([board_HR.shape[1], board_HR.shape[0]])
    target_dim = (LR_dim) // block_size + ((LR_dim % block_size) + block_size - 1) // block_size 
    
    crops = np.array([[board_HR[block_size * j : block_size * (j + 1), block_size * i : block_size * (i + 1)]
                   for i in range(target_dim[0])] 
                   for j in range(target_dim[1])], dtype=object)
    with torch.no_grad():
        torch.cuda.empty_cache()
        degraded_blocks = np.array([[ hr2lr(netG, device, crops[i, j]) 
                                     for i in range(target_dim[1])] 
                                    for j in range(target_dim[0])], dtype=object)
        
    return(concat_vh(degraded_blocks))
    