import nibabel as nib
import scipy
import matplotlib.pyplot as plt

# Naloži datoteko
image = nib.load("../ThoraxCBCT/train/ThoraxCBCT_0000_0002.nii.gz")
data = image.get_fdata()
# image = nib.load("source/Data/image_A.nii.gz")
# data = image.get_fdata()

# Preveri dimenzije
print("Dimenzije slike:", data.shape)

# Spreminjanje velikosti slike na 256x256x256
resized_data = scipy.ndimage.zoom(data, (256/data.shape[0], 256/data.shape[1], 256/data.shape[2]), order=1)

# Preveri nove dimenzije
print("Nove dimenzije slike:", resized_data.shape)

# Izberi srednjo rezino
slice_index = data.shape[1] // 2  # srednja rezina

# Izreži posamezno 2D rezino iz vsake slike
original_slice = data[:, slice_index, :]
resized_slice = resized_data[:, slice_index, :]

# Prikaz posameznih rezin
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Prikaz originalne slike
ax[0].imshow(original_slice.T, cmap='gray', origin='lower')
ax[0].set_title('Original Image')
ax[0].axis('off')

# Prikaz resized slike
ax[1].imshow(resized_slice.T, cmap='gray', origin='lower')
ax[1].set_title('Resized Image')
ax[1].axis('off')

# Pokaži vse slike
plt.tight_layout()
plt.show()