#!/usr/bin/env python3

import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
import glob
import re
import os
import pandas as pd


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s)]

def create_patches(img_path, kh, kw, dh, dw, ch_pt, file_extension = '.npy'):
    """
    Creates a numpy array of patches of a list of images.
    Returns a list of tensors where each element in the list is a tensor of patches from the image in the respective index
    Args:
        root_dir (string): Path to the root folder of all the images that will be used to create the patches.
        kh (int): Kernel Height.
        hw (int): Kernel Width.
        dh (int): Stride Height.
        dw (int): Stride Width.
        ch_pt (int): Amount of bands/channels to take in each patch.
        file_extension (string): File extension in wich are the images saved, for now only supports numpy matrices.
    """
    x = torch.Tensor(np.load(img_path))
    #x = x.reshape(x.shape[2], x.shape[0], x.shape[1])
    
    patches = x.unfold(0, kh, dh).unfold(1, kw, dw).unfold(2, ch_pt, ch_pt)
    #unfold_shape = patches.size()

    patches = patches.contiguous().view(-1, kh, kw, ch_pt)
    
    #patches = x.unfold(2, h_pt, w_pt).unfold(1, h_pt, w_pt)
    #patches = patches.contiguous().view(-1, h_pt, w_pt, ch_pt)

    return patches

def create_csv_patches(root_dir_MSI, root_dir_HSI, h_pt, w_pt, ch_pt_MSI, ch_pt_HSI, stride_h, stride_w, file_extension = ['.npy', '.npy'],
                       start_folder = 1, stop_folder = None,
                       save = False, save_dir_x = None, save_dir_y = None, save_target = True):
    """
    Reads all the images in the directories of the MSI and HSI images, creates the patches for each image,
        saves the patches as pytorch tensors in the 'x' and 'y' directories and returns a csv file with the association between
        each x patch and the corresponding y patch.
        Only returns a pandas DataFrame when save = True, otherwise return two tensors with all the patches of each image.
        
        Args:
            - root_dir_MSI (string): Path to the root folder that contains all the images of the Multi Spectral Images.
            - root_dir_HSI (string): Path to the root folder that contains all the images of the Hyper Spectral Images.
            - h_pt (int): Height of the patches.
            - w_pt (int): Width of the patches.
            - ch_pt_MSI (int): Amount of bands/channels to take in each patch for the Multi Spectral Image.
            - ch_pt_HSI (int): Amount of bands/channels to take in each patch for the Hyper Spectral Image.
            - file_extension (list): list of the file extensions in wich are the images saved for the MSI and HSI respectively,
                for now only supports numpy matrices.
            - start_folder (int): Folder number in which start to create the patches.
            - stop_folder (int): Folder number in which start to create the patches
            - save (bool): Indicates whether the patches must be saved to disk or not.
            - save_dir_x (string): Path to the folder in which the patches of the MSI will be saved if `save = True`.
            - save_dir_y (string): Path to the folder in which the patches of the HSI will be saved if `save = True`.
            - save_target (bool): Default is True, in which case the patches calculated for the target dataset (HSI) will
                be saved in the directory specified by the `save_dir_y`, otherwise not. It is used when creating multiple 
                datasets so you dont have to overwrite the target directory.
    """
    # get the image paths for the MS and HS images in a list, I sort them by alphanumerical order in both cases
    # So no ensure that you will associate the correct images from the MSI and HSI folder I recommend that you save them
    # with the same name in their corresponding folder
    img_paths_MSI = glob.glob(f'{root_dir_MSI}*{file_extension[0]}')
    if stop_folder is None:
        stop_folder = len(img_paths_MSI)
    img_paths_MSI.sort(key=alphanum_key)
    img_paths_MSI = img_paths_MSI[start_folder-1:stop_folder]
    
    img_paths_HSI = glob.glob(f'{root_dir_HSI}*{file_extension[1]}')
    img_paths_HSI.sort(key=alphanum_key)
    img_paths_HSI = img_paths_HSI[start_folder-1:stop_folder]
    
    df_patches = pd.DataFrame(columns = ['Scene_id', 'MSI_path', 'HSI_path'])
    
    if save: # Save the patches as individual tensor files with .pt extension
        """
        if save_dir_x is None:
            print('If save = True save_dir_x must be provided')
            return
        if save_dir_y is None:
            print('If save = True save_dir_y must be provided')
            return 
        try:
            if not os.path.exists(directory_x):
                os.makedirs(directory_x)
            if not os.path.exists(directory_y):
                os.makedirs(directory_y)
        except:
            print('There was an error while creating one of the save directories.')
            return
        """
        for img_MSI, img_HSI in zip(img_paths_MSI, img_paths_HSI): # For each image create the patches
            scene_id = re.match(r'.*\\(\d+).npy', img_MSI).groups()[0]
            print(f'Creating patches of Scene ID: {scene_id}')
            patches_MSI = create_patches(img_MSI, h_pt, w_pt, stride_h, stride_w, ch_pt_MSI)
            if save_target:
                patches_HSI = create_patches(img_HSI, h_pt, w_pt, stride_h, stride_w, ch_pt_HSI)
            

            patch_id = 0
            for msi_patch, hsi_patch in zip(patches_MSI, patches_HSI): # For each patch, save in the directories specified
                msi_file_path = f'{save_dir_x}/{scene_id}_{patch_id}.pt'
                hsi_file_path = f'{save_dir_y}/{scene_id}_{patch_id}.pt'
                
                torch.save(msi_patch, msi_file_path)
                if save_target:
                    torch.save(hsi_patch, hsi_file_path)
                
                patch_id+=1
                
                # Add the row with the association between msi and hsi patches
                df_patches = df_patches.append({'Scene_id': scene_id, 'MSI_path': msi_file_path,
                                  'HSI_path':hsi_file_path}, ignore_index = True)
        df_patches.to_csv('df_patches.csv')
                
        return df_patches
    else: # This simply returns two tensors with size sum(n_patches) x patch_height x patch_width x patch_channels each
        msi_tensor = torch.Tensor()
        hsi_tensor = torch.Tensor()
        for img_MSI, img_HSI in zip(img_paths_MSI, img_paths_HSI): # For each image create the patches
            torch.cat([msi_tensor, create_patches(img_MSI, h_pt, w_pt, stride_h, stride_w, ch_pt_MSI)], out = msi_tensor)
            torch.cat([hsi_tensor, create_patches(img_HSI, h_pt, w_pt, w_pt, stride_h, stride_w, ch_pt_HSI)], out = hsi_tensor)
        return msi_tensor, hsi_tensor

def create_patches_npy(img_path, kh, kw, dh, dw, ch_pt, file_extension = '.npy'):
    """
    Creates a numpy array of patches of a list of images.
    Returns a list of tensors where each element in the list is a tensor of patches from the image in the respective index
    Args:
        root_dir (string): Path to the root folder of all the images that will be used to create the patches.
        kh (int): Kernel Height.
        hw (int): Kernel Width.
        dh (int): Stride Height.
        dw (int): Stride Width.
        ch_pt (int): Amount of bands/channels to take in each patch.
        file_extension (string): File extension in wich are the images saved, for now only supports numpy matrices.
    """
    x = torch.Tensor(np.load(img_path))
    #x = x.reshape(x.shape[2], x.shape[0], x.shape[1])
    
    patches = x.unfold(0, kh, dh).unfold(1, kw, dw).unfold(2, ch_pt, ch_pt)
    #unfold_shape = patches.size()

    patches = patches.contiguous().view(-1, kh, kw, ch_pt)
    
    #patches = x.unfold(2, h_pt, w_pt).unfold(1, h_pt, w_pt)
    #patches = patches.contiguous().view(-1, h_pt, w_pt, ch_pt)

    return np.array(patches)

def create_csv_patches_npy(root_dir_MSI, root_dir_HSI, h_pt, w_pt, ch_pt_MSI, ch_pt_HSI, stride_h, stride_w,
                           file_extension = ['.npy', '.npy'],
                           save = False, save_dir_x = None, save_dir_y = None, save_target = True):
    """
    Reads all the images in the directories of the MSI and HSI images, creates the patches for each image,
        saves the patches as pytorch tensors in the 'x' and 'y' directories and returns a csv file with the association between
        each x patch and the corresponding y patch.
        Only returns a pandas DataFrame when save = True, otherwise return two tensors with all the patches of each image.
        
        Args:
            - root_dir_MSI (string): Path to the root folder that contains all the images of the Multi Spectral Images.
            - root_dir_HSI (string): Path to the root folder that contains all the images of the Hyper Spectral Images.
            - h_pt (int): Height of the patches.
            - w_pt (int): Width of the patches.
            - ch_pt_MSI (int): Amount of bands/channels to take in each patch for the Multi Spectral Image.
            - ch_pt_HSI (int): Amount of bands/channels to take in each patch for the Hyper Spectral Image.
            - file_extension (list): list of the file extensions in wich are the images saved for the MSI and HSI respectively,
                for now only supports numpy matrices.
            - start_folder (int): Folder number in which start to create the patches.
            - stop_folder (int): Folder number in which start to create the patches
            - save (bool): Indicates whether the patches must be saved to disk or not.
            - save_dir_x (string): Path to the folder in which the patches of the MSI will be saved if `save = True`.
            - save_dir_y (string): Path to the folder in which the patches of the HSI will be saved if `save = True`.
            - save_target (bool): Default is True, in which case the patches calculated for the target dataset (HSI) will
                be saved in the directory specified by the `save_dir_y`, otherwise not. It is used when creating multiple 
                datasets so you dont have to overwrite the target directory.
    """
    # get the image paths for the MS and HS images in a list, I sort them by alphanumerical order in both cases
    # So no ensure that you will associate the correct images from the MSI and HSI folder I recommend that you save them
    # with the same name in their corresponding folder
    img_paths_MSI = glob.glob(f'{root_dir_MSI}*{file_extension[0]}')
    img_paths_MSI.sort(key=alphanum_key)
    
    img_paths_HSI = glob.glob(f'{root_dir_HSI}*{file_extension[1]}')
    img_paths_HSI.sort(key=alphanum_key)
    
    df_patches = pd.DataFrame(columns = ['Scene_id', 'MSI_path', 'HSI_path'])
    
    if save: # Save the patches as individual tensor files with .pt extension
        """
        if save_dir_x is None:
            print('If save = True save_dir_x must be provided')
            return
        if save_dir_y is None:
            print('If save = True save_dir_y must be provided')
            return 
        try:
            if not os.path.exists(directory_x):
                os.makedirs(directory_x)
            if not os.path.exists(directory_y):
                os.makedirs(directory_y)
        except:
            print('There was an error while creating one of the save directories.')
            return
        """
        for img_MSI, img_HSI in zip(img_paths_MSI, img_paths_HSI): # For each image create the patches
            scene_id = re.match(r'.*\\(\d+).npy', img_MSI).groups()[0]
            print(f'Creating patches of Scene ID: {scene_id}')
            patches_MSI = create_patches_npy(img_MSI, h_pt, w_pt, stride_h, stride_w, ch_pt_MSI)
            if save_target:
                patches_HSI = create_patches_npy(img_HSI, h_pt, w_pt, stride_h, stride_w, ch_pt_HSI)
            

            patch_id = 0
            for msi_patch, hsi_patch in zip(patches_MSI, patches_HSI): # For each patch, save in the directories specified
                msi_file_path = f'{save_dir_x}/{scene_id}_{patch_id}.npy'
                hsi_file_path = f'{save_dir_y}/{scene_id}_{patch_id}.npy'
                
                np.save(msi_file_path, msi_patch)
                if save_target:
                    np.save(hsi_file_path, hsi_patch)
                
                patch_id+=1
                
                # Add the row with the association between msi and hsi patches
                df_patches = df_patches.append({'Scene_id': scene_id, 'MSI_path': msi_file_path,
                                  'HSI_path':hsi_file_path}, ignore_index = True)
        df_patches.to_csv('df_patches.csv')
        print('Done!')
        return df_patches
    else: # This simply returns two tensors with size sum(n_patches) x patch_height x patch_width x patch_channels each
        msi_tensor = torch.Tensor()
        hsi_tensor = torch.Tensor()
        for img_MSI, img_HSI in zip(img_paths_MSI, img_paths_HSI): # For each image create the patches
            torch.cat([msi_tensor, create_patches(img_MSI, h_pt, w_pt, stride_h, stride_w, ch_pt_MSI)], out = msi_tensor)
            torch.cat([hsi_tensor, create_patches(img_HSI, h_pt, w_pt, w_pt, stride_h, stride_w, ch_pt_HSI)], out = hsi_tensor)
        return msi_tensor, hsi_tensor
        
class SpectralDataset_npy(Dataset):
    def __init__(self, root_dir_multi, root_dir_hyp, transforms = None):
        """
        Args:
            root_dir_multi (string): Path to the root directory of the multispectral images.
            root_dir_hyper (string): Path to the root directory of the hyperspectral images.
            transforms (torch.Compose): Transformations to apply to the data.

        """
        self.transforms = transforms
        self.msi_path_list = glob.glob(root_dir_multi+'/*.npy')
        self.hsi_path_list = glob.glob(root_dir_hyp+'/*.npy')
        if len(self.msi_path_list) != len(self.hsi_path_list):
            print('Different amount of x and y images in the train folders')
            return
        else:
            self.data_len = len(self.msi_path_list)
        
    def __getitem__(self, index):
        img_x = torch.Tensor(np.load(self.msi_path_list[index])).permute(2,0,1) # Permute dims to have CxHxW
        img_y = torch.Tensor(np.load(self.hsi_path_list[index])).permute(2,0,1) # Permute dims to have CxHxW
        
        #print(f'X shape: {img_x.shape}')
        #print(f'X shape: {img_y.shape}')
        
        if self.transforms is not None:
            img_x = self.transforms(img_x)
            img_y = self.transforms(img_y)

        return img_x, img_y

    def __len__(self):
        return self.data_len
