import sys
from codes.GAN_class import DownsampleGAN
from  codes import my_utils
import cv2 as cv
import torch
import os

###
# python3 run.py ./weights/checkpoints/netG_9  ../test1 ../test1_dws
###

if __name__ == "__main__":
    print('torch.cuda.is_available:', torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    pathes = {'generator': sys.argv[1], 'load': sys.argv[2], 'save': sys.argv[3]}
    generator = my_utils.create_generator(pathes['generator'], device)
    #'./weights/checkpoints/netG_9'
    if not os.path.exists(pathes['save']):
        os.mkdir(pathes['save'])
        
    file_names = os.listdir(pathes['load'])    

    file_pathes = [map(lambda name: os.path.join(pathes['load'], name), file_names), 
        map(lambda name: os.path.join(pathes['save'], name), file_names)]

    files = [[], []]
    for file_load, file_save in zip(file_pathes[0], file_pathes[1]):
        if os.path.isfile(file_load): 
            files[0].append(file_load)
            files[1].append(file_save)
    files[0].sort()

    
    for path_load, path_save in zip(files[0], files[1]):
        img = my_utils.read_img(path_load)
        with torch.no_grad():
            torch.cuda.empty_cache() 
            input = img.to(device)
            downsampled_img = my_utils.get_out(generator(input).detach().cpu().numpy()[0])
            cv.imwrite(path_save, downsampled_img) 
    

 