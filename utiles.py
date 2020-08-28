#!/usr/bin/env python3

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Conv2d, BatchNorm2d, MaxPool2d, Conv3d, BatchNorm3d, MaxPool3d, Module, MSELoss, ConvTranspose2d, ConvTranspose3d
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sn
from spectral import *
from metrics import SSIM
import re
import glob


def show_spectral_reconstruction(model, hyper_path, multi_path, pos_row, pos_col, scene_patch, wav_hsi, wav_msi, device = 'cpu'):
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


    sn.lineplot(x = wav_hsi['Wavelength [micro m]'].values, y = patch_hyp[pos_row,pos_col,:]/10000, 
                label = 'True Hyperion Spectra', ax = ax, marker = 'o')
    sn.lineplot(x = wav_hsi['Wavelength [micro m]'].values, y = yhat_cpu[pos_row, pos_col, :]/10000, 
                label = 'Reconstructed Spectra', ax = ax, marker = 'o')
    sn.lineplot(x = wav_msi['Wavelength [micro m]'].values, y = patch_landsat[pos_row, pos_col, :]/10000, marker = "o",
                label = 'Original LandSat-8 OLI Spectra', ax = ax)
    ax.set_xlabel('Wavelength [micro m]', fontsize = 12)
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
def get_hyperion_landsat_bands(hyperion_properties_path, landsat_properties_path, hyperion_imgs_path = 'Images/hyperion_test_npy'):
    """
        Reads a metadata file from Hyperion; LandSat and returns a dataset with the corresponding wavelengths associated with earch band, useful for plotting the spectral reconstruction.
        
    Args:
        - hyperion_imgs_path (string): Path to the patches of hyperion. 
        - hyperion_properties_path (string): The path to the metadata file of any hyperion image, it should be a .txt file.
        - landsat_properties_path (string): The path to the metadata file of any hyperion image, it should be a .txt file.
        
    """
    
    dfprop_Hyperion = pd.DataFrame(columns = ['Band_n', 'Wavelength [micro m]']) 
    regex_wavelenght = re.compile(r'(\d+) = (.*)')
    properties_h = open(hyperion_properties_path, "r")
    regex_wavelenght = re.compile(r'(\d+) = (.*)')
    
    dfprop_Hyperion = pd.DataFrame(columns = ['Band_n', 'Wavelength [micro m]']) 
    
    
    for line in properties_h:
        if 'Wavelengths' in line:
            if len(regex_wavelenght.findall(line)) > 0:
                dfprop_Hyperion = dfprop_Hyperion.append({'Band_n': int(regex_wavelenght.findall(line)[0][0]),
                                       'Wavelength [micro m]': float(regex_wavelenght.findall(line)[0][1])/1000}, ignore_index = True)
    
    
    properties_h = open(landsat_properties_path, "r")
    
    dfprop_L8 = pd.DataFrame(columns = ['Band_n', 'Wavelength [micro m]']) 
    
    for line in properties_h:
        if 'Wavelengths' in line:
            if len(regex_wavelenght.findall(line)) > 0:
                dfprop_L8 = dfprop_L8.append({'Band_n': int(regex_wavelenght.findall(line)[0][0]),
                                       'Wavelength [micro m]': float(regex_wavelenght.findall(line)[0][1])}, ignore_index = True)
    
    dfprop_Hyperion.set_index('Band_n', inplace = True)
    dfprop_L8.set_index('Band_n', inplace = True)
    scenes_patches_list = [re.findall(re.compile(r'\\(.*)\.npy$'), path)[0] for path in glob.glob(f'{hyperion_imgs_path}/*.npy')]
    hyperion_BadBands = list(range(1,8)) + list(range(58,77)) + list(range(225,243)) + list(range(121,127)) +\
                            list(range(167,181)) + list(range(222,242))
    dfprop_Hyperion.drop(hyperion_BadBands, inplace = True)
    dfprop_Hyperion.reset_index(inplace = True)
    
    return (dfprop_Hyperion, dfprop_L8, scenes_patches_list)


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
    val_ssim = []
    train_step = make_train_step(model, loss_fn, optimizer)
    ssim = SSIM()
    
    
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
                val_ssim_batch = []
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    
                    model.eval()
                    
                    yhat = model(x_val).reshape(x_val.shape[0], 175, 64, 64)
                    val_loss = loss_fn(y_val, yhat)
                    val_ssim_batch.append(ssim(y_val, yhat).item())
                    val_losses_batch.append(val_loss.item())
                val_ssim.append(np.mean(val_ssim_batch))
                val_losses.append(np.mean(val_losses_batch))
                print(f'\t  * Val loss: {val_losses[-1]}')
                print(f'\t  * Val SSIM: {val_ssim[-1]}')
    return {'train_losses': losses, 'val_losses': val_losses, 'val_ssim': val_ssim}


###############################################################################################


def get_metrics(model, test_loader, device = 'cpu', train_dict = None, per_band_error = False, prop_hyperion = None):
    test_ssims = []
    test_losses = []
    loss_fn = MSELoss(reduction = 'mean')
    #mse_noReduction = MSELoss(reduction = 'none')
    dict_bands = {}
    ssim = SSIM()


    if per_band_error and (prop_hyperion is not None):
        dict_bands['Coastal Aerosol'] = prop_hyperion[(prop_hyperion['Wavelength [micro m]'] >= 0.43) & (prop_hyperion['Wavelength [micro m]'] <= 0.45)].index.values
        dict_bands['Blue'] = prop_hyperion[(prop_hyperion['Wavelength [micro m]'] >= 0.45) & (prop_hyperion['Wavelength [micro m]'] <= 0.51)].index.values
        dict_bands['Green'] = prop_hyperion[(prop_hyperion['Wavelength [micro m]'] >= 0.53) & (prop_hyperion['Wavelength [micro m]'] <= 0.59)].index.values
        dict_bands['Red'] = prop_hyperion[(prop_hyperion['Wavelength [micro m]'] >= 0.64) & (prop_hyperion['Wavelength [micro m]'] <= 0.67)].index.values
        dict_bands['NIR'] = prop_hyperion[(prop_hyperion['Wavelength [micro m]'] >= 0.85) & (prop_hyperion['Wavelength [micro m]'] <= 0.88)].index.values
        dict_bands['SWIR 1'] = prop_hyperion[(prop_hyperion['Wavelength [micro m]'] >= 1.57) & (prop_hyperion['Wavelength [micro m]'] <= 1.65)].index.values
        dict_bands['SWIR 2'] = prop_hyperion[(prop_hyperion['Wavelength [micro m]'] >= 2.11) & (prop_hyperion['Wavelength [micro m]'] <= 2.29)].index.values
        dict_bands['Panchromatic'] = prop_hyperion[(prop_hyperion['Wavelength [micro m]'] >= 0.50) & (prop_hyperion['Wavelength [micro m]'] <= 0.68)].index.values
        # Con los numeros de banda seleccionar los subtensores de y_test e yhat y meterlo MSE, luego meter los correspondientes valores
        # a una lista asociada a cada banda
    
    
    band_errors = dict([(k, []) for k in dict_bands.keys()])
    
    with torch.no_grad():
        for _, (x_test, y_test) in enumerate(test_loader):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            model.eval()
            
            yhat = model(x_test)
            yhat = yhat.reshape(1,175,64,64)
            
            #yhat2 = yhat.clone().squeeze(0).reshape(175,64,64).permute(1,2,0)
            test_loss = loss_fn(y_test, yhat)
            if per_band_error and prop_hyperion is not None:
                for band in band_errors.keys():
                    #print(y_test[:,dict_bands[band][0]:dict_bands[band][1], :, :].shape)
                    band_errors[band].append(loss_fn(y_test[:, dict_bands[band][0]:dict_bands[band][1], :, :],
                                                     yhat[:, dict_bands[band][0]:dict_bands[band][1], :, :]).item())

            test_losses.append(test_loss.item()) 
            test_ssims.append(ssim(y_test, yhat).item())
  
    print(f'Average MSE (Test): {np.mean(test_losses)}')
    print(f'RMSE (Test): {np.sqrt(np.mean(test_losses))}')
    print(f'RMSE scaled to reflectance (Test): {np.sqrt(np.mean(test_losses))/10000}')
    print(f'Average SSIM (Test): {np.mean(test_ssims)}')
    for band in band_errors.keys():
        print(f'- {band} Band Average RMSE scaled to radiance (Test): {np.sqrt(np.mean(band_errors[band]))/10000}')
    
    if train_dict != None:
        plt.rcParams['figure.figsize'] = (17,10)
        fig, ax1 = plt.subplots(figsize = (17,4))
    
        ax2 = ax1.twinx()
        sn.lineplot(x = range(len(train_dict['train_losses'])), y = train_dict['train_losses'], 
                    label = 'Train MSE per batch', ax = ax1, color = 'dodgerblue')
        sn.lineplot(x = range(len(train_dict['val_losses'])), y = train_dict['val_losses'], 
                    label = 'Validation MSE per batch', ax = ax1, color = 'tomato')
        sn.lineplot(x = range(len(train_dict['val_ssim'])), y = train_dict['val_ssim'], 
                    label = 'Validation SSIM per batch', ax = ax2, color = 'darkgreen')
        sn.despine()
        ax2.legend(loc =  7, fontsize = 12)
        ax1.legend(fontsize = 12)
        plt.plot();
    
    fig,_ = plt.subplots(figsize = (17,10))
    plt.subplot(1, 2, 1)
    sn.distplot(test_ssims)
    plt.axvline(np.mean(test_ssims), label = 'Media del SSIM para Test')
    plt.title('Distribucion de SSIMs (Test)', size = 15)
    plt.legend()
    
    
    plt.subplot(1, 2, 2)
    sn.distplot(test_losses)
    plt.axvline(np.mean(test_losses), label = 'Media del MSE para Test')
    plt.xticks(rotation = 45)
    plt.title('Distribucion de MSE (Test)', size = 15);
    plt.legend()
    
    return {'test_losses': test_losses, 'test_ssims':test_ssims}
                                                                                                                
        

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


class GramLoss(Module):
    def __init__(self, perceptual_model, style_loss = StyleLoss(), content_loss = MSELoss(reduction = 'mean'), contentLoss_coeff = 1,
                styleLoss_coeff = 1e-3, block_coeff = [1, 1, 1], device = 'cpu'):
        super(GramLoss, self).__init__()
        self.perceptual_model = perceptual_model
        self.styleLoss = style_loss
        self.styleLoss.to(device)
        self.contentLoss = content_loss
        self.contentLoss.to(device)
        self.device = device
        self.contentLoss_coeff = contentLoss_coeff
        self.styleLoss_coeff = styleLoss_coeff
        self.block_coeff = block_coeff # es una lista de tres numeros que serán los coeficientes de los bloques
        
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
        style_loss = torch.zeros(1, requires_grad = True, device = self.device)
        #style_loss.to(device)
        for block, coeff in zip(blocks_list, self.block_coeff):
            style((perceptualFeat_gt[block], perceptualFeat_reconstructed[block]))
            style_loss = style_loss + coeff*style.loss # Pondera todas las capas por el correspondiente coeficiente asignado
        
        #print(style_loss.device)
        content_loss = self.contentLoss(img_recon, img_gt)
        #content_loss = Variable(content_loss, requires_grad = True)
        content_loss.requieres_grad = True
        content_loss.to(self.device)
        #print(f'content_loss attached: {content_loss.requires_grad}')
        #print(f'style_loss attached: {style_loss.requires_grad}')
        loss = self.contentLoss_coeff * content_loss + self.styleLoss_coeff * style_loss
        #print(loss)
        loss.to(self.device)
        #print(f'loss attached: {loss.requires_grad}')
        return loss