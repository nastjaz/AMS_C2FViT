import nibabel as nib
import scipy.ndimage

# Naloži datoteko
# image = nib.load("../ThoraxCBCT/imagesTr/ThoraxCBCT_0000_0000.nii.gz")
# data = image.get_fdata()
image = nib.load("source/Data/image_A.nii.gz")
data = image.get_fdata()

# Preveri dimenzije
print("Dimenzije slike:", data.shape)

# Spreminjanje velikosti slike na 128x128x128
resized_data = scipy.ndimage.zoom(data, (256/data.shape[0], 256/data.shape[1], 256/data.shape[2]), order=1)

# Preveri nove dimenzije
print("Nove dimenzije slike:", resized_data.shape)
