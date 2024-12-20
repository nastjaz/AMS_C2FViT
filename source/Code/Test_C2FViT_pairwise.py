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

    moving_img, moving_header, moving_affine = load_4D(moving_path)
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

       # Izračunamo, koliko odrezati na vsaki dimenziji (y in x)
        crop_indices = [(0, target_size[i]) if i == 0 else (0, target_size[i]) for i in range(3)]

        # Izrežemo sliko
        cropped_array = image_array[
            :target_size[0],  # x-dimenzija
            :target_size[1],  # y-dimenzija
            :target_size[2],  # z-dimenzija
        ]

        print(f"Padded size: {padded_size}")
        print(f"Cropping indices: {crop_indices}")
        print(f"Cropped image size: {cropped_array.shape}")
        return cropped_array

        
    def crop_deformation_field(image, target_size=(3, 256, 192, 192)):
        # Preverimo, da je vhodna slika NumPy array
        if not isinstance(image, np.ndarray):
            image_array = np.array(image)
        else:
            image_array = image

        # Velikost originalne slike
        padded_size = image_array.shape

        # Izračunamo, koliko odrezati na vsaki dimenziji (y, x, z)
        crop_indices = [(0, 0), (0, target_size[0]), (0, target_size[1]), (0, target_size[2])]

        # Predpostavljamo, da je image_array oblika (kanali, x, y, z)
        # crop_indices je v obliki [(0, 0), (0, 256), (0, 192), (0, 192)]
        # Izrežemo sliko
        cropped_array = image_array[
            :target_size[0],
            :target_size[1],  # x-dimenzija
            :target_size[2],  # y-dimenzija
            :target_size[3],  # z-dimenzija
        ]


        print(f"Padded size: {padded_size}")
        print(f"Cropping indices: {crop_indices}")
        print(f"Cropped image size: {cropped_array.shape}")
        return cropped_array
   

    with torch.no_grad():
        if com_initial:
            moving_img, init_flow = init_center(moving_img, fixed_img)

        X_down = F.interpolate(moving_img, scale_factor=0.5, mode="trilinear", align_corners=True)
        Y_down = F.interpolate(fixed_img, scale_factor=0.5, mode="trilinear", align_corners=True)

        # warpped_x_list, y_list, affine_para_list = model(X_down, Y_down)
        warpped_x_list, y_list, affine_para_list = model(X_down, Y_down)
        X_Y, affine_matrix = affine_transform(moving_img, affine_para_list[-1])
        # print(len(warpped_x_list))
        # print(warpped_x_list[0].shape)
        # print(warpped_x_list[1].shape)
        # print(warpped_x_list[2].shape)

        # print(len(affine_para_list))
        # print(affine_para_list)


        X_Y_cpu = X_Y.data.cpu().numpy()[0, 0, :, :, :]
        image_shape = X_Y.data.cpu().numpy()[0, 0, :, :, :].shape
        print(f"Velikost slike wraped:{image_shape}")
        

        affine_matrix = affine_matrix.cpu().numpy()
        affine_matrix = np.squeeze(affine_matrix)  # Odstrani nepotrebno zunanjo dimenzijo

        print(f"Afina matrika: {affine_matrix}")
        
        deformation_field_translation = np.ones((256, 256, 256))

        deformation_field_x = np.ones((256, 256, 256)) * affine_matrix[0,3]
        deformation_field_y = np.ones((256, 256, 256)) * affine_matrix[1,3]
        deformation_field_z = np.ones((256, 256, 256)) * affine_matrix[2,3]

        deformation_field_translation = np.array([deformation_field_x, deformation_field_y, deformation_field_z])
        #print(deformation_field_translation)

        # Generiraj mrežo koordinat [x, y, z]
        x = np.arange(image_shape[0])
        y = np.arange(image_shape[1])
        z = np.arange(image_shape[2])
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')

        # Preoblikuj mrežo koordinat v seznam točk
        points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel(), np.ones(x_grid.size)])

        # Transformiraj točke z afino matriko
        transformed_points = affine_matrix @ points  # (3x4) * (4xN)

        # Razdeli transformirane točke na x', y', z'
        x_prime = transformed_points[0, :].reshape(image_shape)
        y_prime = transformed_points[1, :].reshape(image_shape)
        z_prime = transformed_points[2, :].reshape(image_shape)
        # Izpis informacij
        print("Oblika transformiranih koordinat:", x_prime.shape, y_prime.shape, z_prime.shape)
        
        # Izračunaj deformacijsko polje (difference med x', y', z' in originalnimi koordinatami)
        def_field_x = x_prime - x_grid
        def_field_y = y_prime - y_grid
        def_field_z = z_prime - z_grid

        # Oblika deformacijskega polja: [def_field_x, def_field_y, def_field_z]
        deformation_field = np.array([def_field_x, def_field_y, def_field_z])
        
        target_size2 = (256, 192, 192)
        X_Y_cpu = crop_image(X_Y_cpu, target_size=target_size2)

        target_size3 = (3, 256, 192, 192)
        deformation_temp = crop_deformation_field(deformation_field, target_size=target_size3)
        deformation_ = np.transpose(deformation_temp, (1, 2, 3, 0))
        print(deformation_.shape)

        affine_matrix_4x4 = np.eye(4)  # Začetna 4x4 matrika
        affine_matrix_4x4[:3, :] = affine_matrix  # Vnesite svojo 3x4 matriko

        # Ustvarite NIfTI sliko iz deformacijskega polja
        nifti_img = nib.Nifti1Image(deformation_, affine=affine_matrix_4x4)
        print(f"Oblika podatkov v NIfTI sliki: {nifti_img.shape}")

        # Odstranite končnico '.nii.gz' iz obeh
        fixed_base1 = fixed_base.replace('.nii.gz', '').replace('ThoraxCBCT_', '')
        moving_base1 = moving_base.replace('.nii.gz', '').replace('ThoraxCBCT_', '')
        
        # Shrani NIfTI sliko
        output_path = f"{savepath}/disp_{fixed_base1}_{moving_base1}.nii.gz"
        nib.save(nifti_img, output_path)
        
        save_img(X_Y_cpu, f"{savepath}/warped_{fixed_base1}_{moving_base1}.nii.gz", header=header, affine=affine)
        #save_affine_transform(affine_matrix_cpu, f"{savepath}/transform_{moving_base}", header=header, affine=affine)

    print("Result saved to :", savepath)