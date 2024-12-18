import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage
import SimpleITK as sitk

from C2FViT_model import C2F_ViT_stage, AffineCOMTransform, Center_of_mass_initial_pairwise
from Functions import save_img, load_4D, min_max_norm

from UseCuda import device

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--modelpath", type=str,
                        dest="modelpath",
                        default='../Model/C2FViT_affine_COM_pairwise_stagelvl3_118000.pth',
                        help="Pre-trained Model path")
    parser.add_argument("--savepath", type=str,
                        dest="savepath", default='../Result',
                        help="path for saving images")
    parser.add_argument("--fixed", type=str,
                        dest="fixed", default='../Data/image_B.nii.gz',
                        help="fixed image")
    parser.add_argument("--moving", type=str,
                        dest="moving", default='../Data/image_A.nii.gz',
                        help="moving image")
    parser.add_argument("--com_initial", type=bool,
                        dest="com_initial", default=True,
                        help="True: Enable Center of Mass initialization, False: Disable")
    opt = parser.parse_args()

    savepath = opt.savepath
    fixed_path = opt.fixed
    moving_path = opt.moving
    com_initial = opt.com_initial
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    # use_cuda = False
    # # use_cuda = True
    # device = torch.device("cuda" if use_cuda else "cpu")


    model = C2F_ViT_stage(img_size=128, patch_size=[3, 7, 15], stride=[2, 4, 8], num_classes=12,
                          embed_dims=[256, 256, 256],
                          num_heads=[2, 2, 2], mlp_ratios=[2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                          attn_drop_rate=0., norm_layer=nn.Identity,
                          depths=[4, 4, 4], sr_ratios=[1, 1, 1], num_stages=3, linear=False).to(device)

    print(f"Loading model weight {opt.modelpath} ...")
    model.load_state_dict(torch.load(opt.modelpath, map_location=device))
    #model.load_state_dict(torch.load(opt.modelpath))
    model.eval()

    affine_transform = AffineCOMTransform().to(device)
    init_center = Center_of_mass_initial_pairwise()

    fixed_base = os.path.basename(fixed_path)
    moving_base = os.path.basename(moving_path)

    fixed_img_nii = nib.load(fixed_path)
    header, affine = fixed_img_nii.header, fixed_img_nii.affine
    fixed_img = fixed_img_nii.get_fdata()
    fixed_img = np.reshape(fixed_img, (1,) + fixed_img.shape)

    # If fixed img is MNI152 altas, do windowing
    if fixed_base == "MNI152_T1_1mm_brain_pad_RSP.nii.gz":
        fixed_img = np.clip(fixed_img, a_min=2500, a_max=np.max(fixed_img))

    moving_img = load_4D(moving_path)
    print(moving_img.shape)
    print(fixed_img.shape)

    fixed_img = min_max_norm(fixed_img)
    moving_img = min_max_norm(moving_img)

    def add_padding_to_image(image, target_size=(1, 256, 256, 256), padding_value=0):
        if not isinstance(image, np.ndarray):
            image_array = np.array(image)
        else:
            image_array = image

        # Velikost originalne slike
        original_size = image_array.shape

        # Izračun paddinga samo na koncu dimenzij
        padding = [(0, max(target_size[i] - original_size[i], 0)) for i in range(len(original_size))]
    
        # Dodaj padding za dodatne dimenzije, če target_size vsebuje več dimenzij
        if len(target_size) > len(original_size):
            padding += [(0, 0)] * (len(target_size) - len(original_size))
        
        print(f"Original size: {original_size}")
        print(f"Target size: {target_size}")
        print(f"Padding to add: {padding}")

        # Dodajanje paddinga
        padded_array = np.pad(image_array, pad_width=padding, mode='constant', constant_values=padding_value)

        print(f"Padded image shape: {padded_array.shape}")
        return padded_array

    target_size = (1, 256, 256, 256)
    fixed_img = add_padding_to_image(fixed_img, target_size=target_size)
    moving_img = add_padding_to_image(moving_img, target_size=target_size)

    fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
    moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)

    def crop_image(image, target_size=(256, 192, 192)):
        # Preverimo, da je vhodna slika NumPy array
        if not isinstance(image, np.ndarray):
            image_array = np.array(image)
        else:
            image_array = image

        # Velikost originalne slike
        padded_size = image_array.shape

       # Izračun začetnih in končnih indeksov za crop v vsaki dimenziji
        crop_indices = [
            (0, min(target_size[i], padded_size[i]))  # Ohranimo samo del, ki ustreza ciljni velikosti
            for i in range(len(target_size))
        ]

        # Dodamo preostale dimenzije, če jih target_size nima
        crop_indices += [(0, padded_size[i]) for i in range(len(target_size), len(padded_size))]

        print(f"Padded size: {padded_size}")
        print(f"Cropping indices: {crop_indices}")

        # Izrežemo sliko
        cropped_array = image_array[
            crop_indices[0][0]:crop_indices[0][1],
            crop_indices[1][0]:crop_indices[1][1],
            crop_indices[2][0]:crop_indices[2][1],
        ]
    
        print(f"Cropped image size: {cropped_array.shape}")
        return cropped_array

    with torch.no_grad():
        if com_initial:
            moving_img, init_flow = init_center(moving_img, fixed_img)

        X_down = F.interpolate(moving_img, scale_factor=0.5, mode="trilinear", align_corners=True)
        Y_down = F.interpolate(fixed_img, scale_factor=0.5, mode="trilinear", align_corners=True)

        warpped_x_list, y_list, affine_para_list = model(X_down, Y_down)
        X_Y, affine_matrix = affine_transform(moving_img, affine_para_list[-1])

        X_Y_cpu = X_Y.data.cpu().numpy()[0, 0, :, :, :]
        #affine_matrix_cpu = affine_matrix.cpu()
        print(affine_matrix.cpu().numpy().shape)
        print(affine_matrix.cpu().numpy())
        #affine_matrix_cpu = affine_matrix.cpu().numpy()[0, 0, :, :, :]

        target_size2 = (256, 192, 192)
        X_Y_cpu = crop_image(X_Y_cpu, target_size=target_size2)
        #affine_matrix_cpu = 
        save_img(X_Y_cpu, f"{savepath}/warped_{moving_base}", header=header, affine=affine)
        #save_affine_transform(affine_matrix_cpu, f"{savepath}/transform_{moving_base}", header=header, affine=affine)

    print("Result saved to :", savepath)