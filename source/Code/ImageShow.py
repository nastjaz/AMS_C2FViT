import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from nibabel.processing import resample_from_to
import os

# Pot do tvojih .nii.gz datotek
fixed_path = r'C:\Users\Uporabnik\OneDrive - Univerza v Ljubljani\Documents\3. semester\Analiza medicinskih slik\Izziv\AMS_C2FViT\source\Data\image_A.nii.gz'
moving_path = r'C:\Users\Uporabnik\OneDrive - Univerza v Ljubljani\Documents\3. semester\Analiza medicinskih slik\Izziv\AMS_C2FViT\source\Data\image_B.nii.gz'
warped_path = r'C:\Users\Uporabnik\OneDrive - Univerza v Ljubljani\Documents\3. semester\Analiza medicinskih slik\Izziv\Result\warped_image_A.nii.gz'

# Preveri, ali poti obstajajo
if not os.path.exists(fixed_path):
    print(f"Pot do fiksne slike ni pravilna: {fixed_path}")
if not os.path.exists(moving_path):
    print(f"Pot do premikajoče slike ni pravilna: {moving_path}")
if not os.path.exists(warped_path):
    print(f"Pot do poravnane slike ni pravilna: {warped_path}")

# Naloži slike
try:
    fixed_img_nii = nib.load(fixed_path)
    moving_img_nii = nib.load(moving_path)
    warped_img_nii = nib.load(warped_path)
except Exception as e:
    print(f"Napaka pri nalaganju slike: {e}")
    raise

# Poravnaj slike na podlagi "fixed" slike
aligned_moving_img = resample_from_to(moving_img_nii, fixed_img_nii)
aligned_warped_img = resample_from_to(warped_img_nii, fixed_img_nii)

# Pretvori podatke v numpy array
fixed_img = fixed_img_nii.get_fdata()
aligned_moving_img_data = aligned_moving_img.get_fdata()
aligned_warped_img_data = aligned_warped_img.get_fdata()

# Preveri, da imajo vse slike enako število dimenzij
if len(fixed_img.shape) != 3 or len(aligned_moving_img_data.shape) != 3 or len(aligned_warped_img_data.shape) != 3:
    print("Ena ali več slik nima 3D dimenzij.")
    raise ValueError("Vse slike morajo biti 3D.")

# Izberi srednjo rezino
slice_index = fixed_img.shape[2] // 2  # srednja rezina

# Izreži posamezno 2D rezino iz vsake slike
fixed_slice = fixed_img[:, :, slice_index]
moving_slice = aligned_moving_img_data[:, :, slice_index]
warped_slice = aligned_warped_img_data[:, :, slice_index]

# Prikaz posameznih rezin
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Prikaz "fixed" slike
ax[0].imshow(fixed_slice.T, cmap='gray', origin='lower')
ax[0].set_title('Fixed Image')
ax[0].axis('off')

# Prikaz "moving" slike
ax[1].imshow(moving_slice.T, cmap='gray', origin='lower')
ax[1].set_title('Moving Image')
ax[1].axis('off')

# Prikaz "warped" (poravnane) slike
ax[2].imshow(warped_slice.T, cmap='gray', origin='lower')
ax[2].set_title('Warped Image')
ax[2].axis('off')

# Pokaži vse slike
plt.tight_layout()
plt.show()
