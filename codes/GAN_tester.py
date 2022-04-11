import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset, random_split
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid
from torchvision.utils import save_image
import torch.optim as optim
from codes.GAN_class import DownsampleGAN
from codes import my_utils 
from tqdm import tqdm
import numpy as np

class TestGAN():   

    def __init__(self,  model: DownsampleGAN):
        self.generator = model.generator
        self.discriminator = model.discriminator
        self.validation_loader = model.validation_loader
        self.device = model.device
        self.criterion = model.criterion
        self.criterionG = model.criterionG
        self.model

    def test(self, epoch):
        for data in self.validation_loader.dataset:
            torch.cuda.empty_cache()
            img_GT = np.array(data['GT'])
            img_LQ = torch.tensor(np.array(data['LQ']))

            HR = torch.tensor(my_utils.get_in(img_GT)).float()
            HR = HR.resize(1, HR.size(0), HR.size(1), HR.size(2))

            
            LR = torch.tensor(my_utils.get_in(img_LQ)).float()
            LR = LR.resize(1, LR.size(0), LR.size(1), LR.size(2))
            
            self.discriminator.zero_grad()
            real_img = LR.to(self.device)
            #print(real_img.size())
            real_D = self.discriminator(real_img).view(-1)
            label = torch.ones(real_D.size(), dtype=torch.float, device = self.device) 
            errD_real = self.criterion(real_D, label)
            #errD_real.backward()
            D_x = real_D.mean().item()

            #HR = add_noise(HR, img_size) 
            HR = my_utils.dd_noise(HR, [HR.size(2), HR.size(3)])
            fake = self.generator(HR.to(self.device))
            label.fill_(0)
            fake_D = self.discriminator(fake.detach()).view(-1)
            errD_fake = self.criterion(fake_D, label)  
            #errD_fake.backward()
            D_G_z1 = fake_D.mean().item()
            errD = errD_real + errD_fake
            #optimizerD.step()


            self.generator.zero_grad()
            label.fill_(1) 
            fake_D = self.discriminator(fake).view(-1)

            errG_mse = self.criterionG(fake, real_img)
            errG = self.criterion(fake_D, label) + mse_weight * errG_mse 
            #errG.backward()
            D_G_z2 = fake_D.mean().item()
            #optimizerG.step()

            error_G += errG.item()
            error_D += errD.item()
            
            error_G /= len(self.validation_loader)
            error_D /= len(self.validation_loader)
            V_lss['G'].append(error_G) 
            V_lss['D'].append(error_D)
            print('Test Gen loss after %d epoch = ' % int(epoch + 1), error_G)
            print('Test Dis loss after %d epoch = ' % int(epoch + 1), error_D)
            print('################### --- epoch ' + str(epoch + 1) + ' finished --- ###################')        












'''

error_G = 0
        error_D = 0
        for data in val_loader.dataset:
            torch.cuda.empty_cache()
            img_GT = np.array(data['GT'])
            img_LQ = torch.tensor(np.array(data['LQ']))

            HR = torch.tensor(get_in(img_GT)).float()
            #inp = torch.tensor(img_GT).float()
            HR = HR.resize(1, HR.size(0), HR.size(1), HR.size(2))
            #print(HR.size())
            #HR = add_noise(inp, [inp.size(2), inp.size(3)]).to(device)
            #val_loss += crit(get_out(netG(inp)[0]), img_LQ.to(device)).item()
            
            LR = torch.tensor(get_in(img_LQ)).float()
            LR = LR.resize(1, LR.size(0), LR.size(1), LR.size(2))
            
            netD.zero_grad()
            real_img = LR.to(device)
            #print(real_img.size())
            real_D = netD(real_img).view(-1)
            label = torch.ones(real_D.size(), dtype=torch.float, device = device) 
            errD_real = criterion(real_D, label)
            #errD_real.backward()
            D_x = real_D.mean().item()

            #HR = add_noise(HR, img_size) 
            HR = add_noise(HR, [HR.size(2), HR.size(3)])
            fake = netG(HR.to(device))
            label.fill_(0)
            fake_D = netD(fake.detach()).view(-1)
            errD_fake = criterion(fake_D, label)  
            #errD_fake.backward()
            D_G_z1 = fake_D.mean().item()
            errD = errD_real + errD_fake
            #optimizerD.step()


            netG.zero_grad()
            label.fill_(1) 
            fake_D = netD(fake).view(-1)

            errG_mse = criterionG(fake, real_img)
            errG = criterion(fake_D, label) + mse_weight * errG_mse 
            #errG.backward()
            D_G_z2 = fake_D.mean().item()
            #optimizerG.step()

            error_G += errG.item()
            error_D += errD.item()
            
        error_G /= len(val_loader)
        error_D /= len(val_loader)
        V_lss['G'].append(error_G) 
        V_lss['D'].append(error_D)
        print('Test Gen loss after %d epoch = ' % int(epoch + 1), error_G)
        print('Test Dis loss after %d epoch = ' % int(epoch + 1), error_D)
        print('################### --- epoch ' + str(epoch + 1) + ' finished --- ###################')
'''