import nibabel as nib

# Naloži datoteko
# image = nib.load("../ThoraxCBCT/imagesTr/ThoraxCBCT_0000_0000.nii.gz")
# data = image.get_fdata()
image = nib.load("source/Data/image_A.nii.gz")
data = image.get_fdata()

# Preveri dimenzije
print("Dimenzije slike:", data.shape)
