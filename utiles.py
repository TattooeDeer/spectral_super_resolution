#!/usr/bin/env python3

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Conv2d, BatchNorm2d, MaxPool2d, Conv3d, BatchNorm3d, MaxPool3d, Module, MSELoss, ConvTranspose2d, ConvTranspose3d
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sn
from spectral import *


def show_spectral_reconstruction(model, hyper_path, multi_path, pos_row, pos_col, scene_patch, wav_hsi, wav_msi):
    """
    Shows the spectral reconstruction accomplished by a model selecting a random pixel position and a random scene/patch, for this, a lineplot of a 
    alongside the  
    
    Args:
        - model (torch model): Model to use to predict the spectra.
        - hyper_path (string): Folder path to the hyper spectral images to show.
        - multi_path (string): Folder path to the multi spectral images to show.
        - pos_row (int): Integer referencing the row position of the pixel to show.
        - pos_col (int): Integer referencing the col position of the pixel to show.
        - scene_patch (string): String referencing the name of the Scene Id and the patch to load.
        - wav_hsi (DataFrame): Pandas DataFrame with at least one column named 'Wavelength' that contains the 
    
    Return value:
        Void function. Plots a lineplot of the 
    """
    
    patch_hyp = torch.Tensor(np.load(f'{hyper_path}/{scene_patch}.npy'))
    patch_landsat = torch.Tensor(np.load(f'{multi_path}/{scene_patch}.npy'))
    
    patch_landsat_reshaped = patch_landsat.unsqueeze(0).permute(0,3,1,2).to(device) 
    model.eval()

    yhat = model(patch_landsat_reshaped).squeeze(0).reshape(175,64,64).permute(1,2,0) #queda lista
    yhat_cpu = yhat.to('cpu').detach().numpy()
        #
    fig, ax = plt.subplots(figsize = (16,5))
    fig.suptitle(f'Scene {scene_patch}', fontsize=16)


    sn.lineplot(x = wav_hsi.Wavelength.values, y = patch_hyp[pos_row,pos_col,:]/10000, 
                label = 'True Hyperion Spectra', ax = ax, marker = 'o')
    sn.lineplot(x = wav_hsi.Wavelength.values, y = yhat_cpu[pos_row, pos_col, :]/10000, 
                label = 'Reconstructed Spectra', ax = ax, marker = 'o')
    sn.lineplot(x = wav_msi.Wavelength.values, y = patch_landsat[pos_row, pos_col, :]/10000, marker = "o",
                label = 'Original LandSat-8 OLI Spectra', ax = ax)
    ax.set_xlabel('Wavelength [nm]', fontsize = 12)
    ax.set_ylabel('Reflectance (prop.)', fontsize = 12)
    sn.despine()
    
    fig, ax2 = plt.subplots(1,2,figsize = (16,5))
    sn.distplot((np.array(patch_hyp[:,:,29])/10000).flatten(), ax = ax2[0])
    sn.distplot((np.array(patch_hyp[:,:,20])/10000).flatten(), ax = ax2[0])
    sn.distplot((np.array(patch_hyp[:,:,12])/10000).flatten(), ax = ax2[0]);
    
    sn.distplot(yhat_cpu[:,:,29].flatten()/10000, ax = ax2[1])
    sn.distplot(yhat_cpu[:,:,20].flatten()/10000, ax = ax2[1])
    sn.distplot(yhat_cpu[:,:,12].flatten()/10000, ax = ax2[1]);
    sn.despine()
    
    plt.show()
    ax.legend(fontsize = 10)

    imshow(yhat_cpu/10000, (29,20,12))
    plt.axvline(pos_row, color = 'red')
    plt.axhline(63-pos_col, color = 'red')

    imshow(np.array(patch_hyp)/10000, (29,20,12))
    plt.axvline(pos_row, color = 'red')
    plt.axhline(63-pos_col, color = 'red')


## Codigo para sacar las longitudes de onda de hyperion
def get_hyperion_landsat_bands(hyperion_properties_path = 'Images/propertiesH_metadata.txt', landsat_properties_path = "Images/propertiesL8_metadata.txt"):
    """
        Reads a metadata file from Hyperion; LandSat and returns a dataset with the corresponding wavelengths associated with earch band, useful for plotting the spectral reconstruction.
        
    Args:
        - hyperion_properties_path (string): The path to the metadata file of any hyperion image, it should be a .txt file.
        - landsat_properties_path (string): The path to the metadata file of any hyperion image, it should be a .txt file.
        
    """
    
    dfprop_Hyperion = pd.DataFrame(columns = ['Band_n', 'Wavelength']) 
    regex_wavelenght = re.compile(r'(\d+) = (.*)')
    properties_h = open(hyperion_properties_path, "r")
    regex_wavelenght = re.compile(r'(\d+) = (.*)')
    
    dfprop_Hyperion = pd.DataFrame(columns = ['Band_n', 'Wavelength']) 
    
    
    for line in properties_h:
        if 'Wavelengths' in line:
            if len(regex_wavelenght.findall(line)) > 0:
                dfprop_Hyperion = dfprop_Hyperion.append({'Band_n': int(regex_wavelenght.findall(line)[0][0]),
                                       'Wavelength': float(regex_wavelenght.findall(line)[0][1])/1000}, ignore_index = True)
    
    
    properties_h = open(landsat_properties_path, "r")
    
    dfprop_L8 = pd.DataFrame(columns = ['Band_n', 'Wavelength']) 
    
    for line in properties_h:
        if 'Wavelengths' in line:
            if len(regex_wavelenght.findall(line)) > 0:
                dfprop_L8 = dfprop_L8.append({'Band_n': int(regex_wavelenght.findall(line)[0][0]),
                                       'Wavelength': float(regex_wavelenght.findall(line)[0][1])}, ignore_index = True)
    
    dfprop_Hyperion.set_index('Band_n', inplace = True)
    dfprop_L8.set_index('Band_n', inplace = True)
    scenes_patches_list = [re.findall(r'\\(.*)\.npy', path)[0] for path in glob.glob('Images/hyperion_test_npy/*.npy')]
    hyperion_BadBands = list(range(1,8)) + list(range(58,77)) + list(range(225,243)) + list(range(121,127)) +\
                            list(range(167,181)) + list(range(222,225))
    dfprop_Hyperion.drop(hyperion_BadBands, inplace = True)
    
    return (dfprop_Hyperion, dfprop_L8)


###############################################################################################
def make_train_step(model, loss_fn, optimizer):
    """ Builds a function to perform a single step in the training loop"""
    
    def train_step(x,y):
        model.train() # Set model to train mode
        
        yhat_pre = model(x)
        batch_size = yhat_pre.shape[0]# Make prediction
        yhat = yhat_pre.reshape(batch_size, 175, 64, 64)
        
        loss = loss_fn(yhat, y) # Computes loss

        loss.backward() # Computes gradients
        
        optimizer.step() # Update parameters
        optimizer.zero_grad() # Zeroes the gradients
        
        return loss.item()

    return train_step

def train_model(model, n_epochs, loss_fn, optimizer, train_loader, val_loader, device = 'cpu'):
    # Base hiperparameters for the training
    losses = []
    #psnr_train = []
    val_losses = []
    train_step = make_train_step(model, loss_fn, optimizer)
    
    
    for epoch in range(n_epochs):
        print(f'Epoch N°: {epoch+1}')
        for batch_n, (x_batch, y_batch) in enumerate(train_loader):
            print(f'\tBatch N°: {batch_n}')
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = train_step(x_batch, y_batch)
            losses.append(loss)
            print(f'\t  * Train loss: {losses[-1]}')
            
            with torch.no_grad():
                val_losses_batch = []
                val_psnr_batch = []
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    
                    model.eval()
                    
                    yhat = model(x_val).reshape(x_val.shape[0], 175, 64, 64)
                    val_loss = loss_fn(y_val, yhat)
                    val_losses_batch.append(val_loss.item())
                val_losses.append(np.mean(val_losses_batch))
                print(f'\t  * Val loss: {val_losses[-1]}')
    return {'train_losses': losses, 'val_losses': val_losses}


###############################################################################################


def get_metrics(model, test_loader, device = 'cpu', losses_train = None):
    # No prints
    test_ssims = []
    test_losses = []
    loss_fn = MSELoss(reduction = 'mean')
    n_plots = 3 if losses_train != None else 2
    ssim = SSIM()
    
    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(test_loader):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            model.eval()
            
            yhat = model(x_test)
            yhat = yhat.reshape(1,175,64,64)
            test_ssim = ssim(y_test, yhat)
            test_ssims.append(test_ssim.item())
    
    
    
    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(test_loader):
            if i == 21:
                break
            x_test = x_test.to(device)
            y_test = y_test.to(device).squeeze(0).reshape(175,64,64).permute(1,2,0)
            
            model.eval()
            
            yhat = model(x_test).squeeze(0).reshape(175,64,64).permute(1,2,0) #queda lista
            test_loss = loss_fn(y_test, yhat)
            test_losses.append(test_loss.item())
    
    print(f'Average MSE (Test): {np.mean(test_losses)}')
    print(f'RMSE (Test): {np.sqrt(np.mean(test_losses))}')
    print(f'RMSE scaled to reflectance (Test: {np.sqrt(np.mean(test_losses))/10000}')
    print(f'Average SSIM(Test): {np.mean(test_ssims)}')
    
    
    plt.rcParams['figure.figsize'] = (12,10)
    plt.subplot(1, n_plots, 1)
    sn.distplot(test_ssims)
    plt.title('Distribucion de SSIMs (Test', size = 15)
    
    
    plt.subplot(1, n_plots, 2)
    sn.distplot(test_losses)
    plt.xticks(rotation = 45)
    plt.title('Distribucion de MSE (Test)', size = 15);
    
    
    if losses_train != None:
        plt.subplot(1, n_plots, 3)
        sn.distplot(losses_train)
        plt.xticks(rotation = 45)
        plt.title('Distribucion de MSE (Train)', size = 15);
        

def get_activation(name, activation_dict):
    def hook(model, input, output):
        activation_dict[name] = output.detach()
    return hook

class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
        #self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G_gt = gram_matrix(input[0])
        G_reconstructed = gram_matrix(input[1])
        self.loss = F.mse_loss(G_gt, G_reconstructed)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def get_style_model_and_losses(model, hsi_img, style_layers=['encoder_upperBlock', 'encoder_middleBlock', 'encoder_lowerBlock']):
    cnn = copy.deepcopy(model)
    model = nn.Sequential()
    
    # just in order to have an iterable access to or list of content/syle
    # losses
    style_losses = []

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    target_feature = model(style_img).detach()
    style_loss = StyleLoss(target_feature)
    model.add_module("style_loss_{}".format(i), style_loss)
    style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

#def add_noise(img,noise_type="gaussian"):
#    row,col=64,64
#    img=img.astype(np.float32)
#    
#    if noise_type=="gaussian":
#        mean=0
#        var=10
#        sigma=var**.5
#        noise=np.random.normal(-5.9,5.9,img.shape)
#        #noise=noise.reshape(row,col)
#        img=img+noise
#        return img
#
#    if noise_type=="speckle":
#        noise=np.random.randn(row,col)
#        noise=noise.reshape(row,col)
#        img=img+img*noise
#        return img
#
#class SpectralDataset_Noised(torch.Dataset):
#    def __init__(self, root_dir_multi, root_dir_hyp, transforms = None, percent_noise = 1):
#        """
#        Args:
#            root_dir_multi (string): Path to the root directory of the multispectral images.
#            root_dir_hyper (string): Path to the root directory of the hyperspectral images.
#            transforms (torch.Compose): Transformations to apply to the data.
#
#        """
#        self.noise = torch.randn(self.data.shape)*10000
#        self.percent_noise = percent_noise
#        self.transforms = transforms
#        self.msi_path_list = glob.glob(root_dir_multi+'/*.npy')
#        self.hsi_path_list = glob.glob(root_dir_hyp+'/*.npy')
#        if len(self.msi_path_list) != len(self.hsi_path_list):
#            print('Different amount of x and y images in the train folders')
#            return
#        else:
#            self.data_len = len(self.msi_path_list)
#        
#    def __getitem__(self, index):
#        img_x = torch.Tensor(np.load(self.msi_path_list[index])).permute(2,0,1) # Permute dims to have CxHxW
#        img_y = torch.Tensor(np.load(self.hsi_path_list[index])).permute(2,0,1) # Permute dims to have CxHxW
#        
#        #print(f'X shape: {img_x.shape}')
#        #print(f'X shape: {img_y.shape}')
#        
#        if self.transforms is not None:
#            img_x = self.transforms(img_x)
#            img_y = self.transforms(img_y)
#        noise = self.noise[index]
#        img_x = noise*self.percent_noise + x*(1-self.percent_noise
#                                             )
#        return img_x, img_y
#
#    def __len__(self):
#        return self.data_len
#
#    #
class GramLoss(Module):
    def __init__(self, perceptual_model, style_loss = StyleLoss(), content_loss = MSELoss(reduction = 'mean')):
        super(GramLoss, self).__init__()
        self.perceptual_model = perceptual_model
        self.styleLoss = style_loss
        self.styleLoss.to(device)
        self.contentLoss = content_loss
        self.contentLoss.to(device)
        
    def forward(self, img_recon, img_gt):
        perceptualFeat_gt = {}
        perceptualFeat_reconstructed = {}
        style_losses = []
        blocks_list = ['encoder_upperBlock', 'encoder_middleBlock', 'encoder_lowerBlock']
        
        # Capturar perceptual GT y meter a dict gt
        hook_ub = self.perceptual_model.encoder_upperBlock.register_forward_hook(get_activation('encoder_upperBlock', perceptualFeat_gt))
        hook_mb = self.perceptual_model.encoder_middleBlock.register_forward_hook(get_activation('encoder_middleBlock', perceptualFeat_gt))
        hook_lb = self.perceptual_model.encoder_lowerBlock.register_forward_hook(get_activation('encoder_lowerBlock', perceptualFeat_gt))
        
        # Hacer forward y meter a dict
        self.perceptual_model(img_gt)
        
        #print(perceptualFeat_gt)
        hook_ub.remove()
        hook_mb.remove()
        hook_lb.remove()
        
        # Capturar perceptual recon y meter a dict recon
        hook_ub = self.perceptual_model.encoder_upperBlock.register_forward_hook(get_activation('encoder_upperBlock', perceptualFeat_reconstructed))
        hook_mb = self.perceptual_model.encoder_middleBlock.register_forward_hook(get_activation('encoder_middleBlock', perceptualFeat_reconstructed))
        hook_lb = self.perceptual_model.encoder_lowerBlock.register_forward_hook(get_activation('encoder_lowerBlock', perceptualFeat_reconstructed))
        
        # Hacer forward y meter a dict
        self.perceptual_model(img_recon)
        hook_ub.remove()
        hook_mb.remove()
        hook_lb.remove()
        
        
        style = self.styleLoss
        
        # Calcular las losses
        style_loss = torch.zeros(1, requires_grad = True, device = device)
        #style_loss.to(device)
        for block in blocks_list:
            style((perceptualFeat_gt[block], perceptualFeat_reconstructed[block]))
            style_loss = style_loss + style.loss # Pondera todas las capas en 1
        
        #print(style_loss.device)
        content_loss = self.contentLoss(img_recon, img_gt)
        #content_loss = Variable(content_loss, requires_grad = True)
        content_loss.requieres_grad = True
        content_loss.to(device)
        #print(f'content_loss attached: {content_loss.requires_grad}')
        #print(f'style_loss attached: {style_loss.requires_grad}')
        loss = 1*content_loss + 1e-3*style_loss
        #print(loss)
        loss.to(device)
        #print(f'loss attached: {loss.requires_grad}')
        return loss
    
        