import sys
#from codes.GAN_class import DownsampleGAN
from  codes import my_utils
import cv2 as cv
import torch
import os
from tqdm import tqdm

###
# python3 run.py ./saved_weights/models/mod_19  [inputs folder path] [output folder path] [size of block to cut the image; default 256]
# python3 run.py ./saved_weights/models/mod_19  ../test1/result_parts/part1 ../test1/result_parts_dws/part1 1024
###

if __name__ == "__main__":
    print('torch.cuda.is_available:', torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    pathes = {'model': sys.argv[1], 'load': sys.argv[2], 'save': sys.argv[3]}
    block_size = 256 if not sys.argv[4] else int(sys.argv[4])
    #generator = my_utils.create_generator(pathes['generator'], device)
    dwsGAN = my_utils.load_model(pathes['model'])

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
    files[1].sort()
    
    print('Start image degrading ...')
    
    count = 0	

    for path_load, path_save in tqdm(zip(files[0], files[1])):
        with torch.no_grad():
            torch.cuda.empty_cache() 
            downsampled_img = my_utils.frame_processing(dwsGAN.generator, dwsGAN.device, path_load, block_size=block_size)
            cv.imwrite(path_save, downsampled_img[..., ::-1] * 255) 
        count += 1
        #break #to test single frame

    print(f'Successfully processed {count} frames')        

 
