import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

def resize_image(image, target_size=(256, 256, 256)):
    
    # Pridobi trenutne dimenzije slike in spacing
    original_size = np.array(image.GetSize())
    original_spacing = np.array(image.GetSpacing())

    # Izračun novega spacinga na podlagi tarčne velikosti
    new_spacing = original_spacing * (original_size / np.array(target_size))

    # Resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing.tolist())
    resample.SetSize(target_size)
    resample.SetInterpolator(sitk.sitkLinear)  # Linearna interpolacija
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())

    # Resizaj sliko
    return resample.Execute(image)

# Pot do tvoje NIfTI datoteke
input_filepath = "imagesTr/ThoraxCBCT_0000_0000.nii.gz"
output_filepath = "imagesTr_resized/ThoraxCBCT_0000_0000.nii.gz"

# Nalaganje slike
data = sitk.ReadImage(input_filepath)

# Preveri dimenzije
print("Dimenzije slike:", data.GetSize())

# Resizanje na 256 × 256 × 256
resized_data = resize_image(data, target_size=(256, 256, 256))

# Preveri nove dimenzije
print("Nove dimenzije slike:", resized_data.GetSize())

import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

# Pretvorba SimpleITK slik v Numpy array
original_numpy = sitk.GetArrayFromImage(data)  # Oblika: (z, y, x)
resized_numpy = sitk.GetArrayFromImage(resized_data)

# Izberi indeks rezine za prikaz
slice_index = 100  # Poljubno izbrana rezina, preveri, da je znotraj razpona

# Izreži koronalno rezino
original_slice = original_numpy[:, slice_index, :]  # Dimenzije: (z, x)
resized_slice = resized_numpy[:, slice_index, :]

# Prikaz posameznih rezin
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Prikaz originalne slike
ax[0].imshow(original_slice, cmap='gray', origin='lower')
ax[0].set_title('Original Image')
ax[0].axis('off')

# Prikaz resized slike
ax[1].imshow(resized_slice, cmap='gray', origin='lower')
ax[1].set_title('Resized Image')
ax[1].axis('off')

# Pokaži vse slike
plt.tight_layout()
plt.show()
