import sys
import argparse
from  codes import my_utils
import cv2 as cv
import torch
import os
from tqdm import tqdm


if __name__ == "__main__":
    print('torch.cuda.is_available:', torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='saved_weights/models/mod_19', help='Folder path to the pre-trained models')
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input folder')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-b', '--block_size', type=int, default=128, help='Size of block to split the image')
    
    args = parser.parse_args()

    dwsGAN = my_utils.load_model(args.model)

    if not os.path.exists(args.output):
        os.mkdir(args.output)
        
    file_names = os.listdir(args.input)    

    file_pathes = [map(lambda name: os.path.join(args.input, name), file_names), 
        map(lambda name: os.path.join(args.output, name), file_names)]

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
            downsampled_img = my_utils.frame_processing(dwsGAN.generator, dwsGAN.device, path_load, block_size=args.block_size)
            cv.imwrite(path_save, downsampled_img[..., ::-1] * 255) 
        count += 1
        #break #to test single frame

    print(f'Successfully processed {count} frames')        

 
