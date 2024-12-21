import os
import SimpleITK as sitk
import numpy as np

def add_padding_to_image(image, target_size=(256, 256, 256), padding_value=0):
    # Pretvori sliko v NumPy array
    image_array = sitk.GetArrayFromImage(image)
    
    # Prvotna velikost
    original_size = image_array.shape  # Oblika v (z, y, x)
    
    # Izračun paddinga
    padding = [(0, target_size[i] - original_size[i]) for i in range(3)]
    
    # Dodajanje paddinga
    padded_array = np.pad(image_array, pad_width=padding, mode='constant', constant_values=padding_value)

    # Pretvorba nazaj v SimpleITK
    final_image = sitk.GetImageFromArray(padded_array)
    final_image.SetSpacing(image.GetSpacing())  # Ohranimo spacing
    final_image.SetOrigin(image.GetOrigin())    # Ohranimo origin
    final_image.SetDirection(image.GetDirection())  # Ohranimo direction
    return final_image

# Pot do map slik
input_dir_images = "/media/FastDataMama/nastjaz/AMS_C2FViT/source/OriginalData"
output_dir_images = "/media/FastDataMama/nastjaz/AMS_C2FViT/source/Data"
validation_dir_images = os.path.join(output_dir_images, "validation")

# Pot do map labels
input_dir_labels = "/media/FastDataMama/nastjaz/AMS_C2FViT/source/OriginalLabels"
output_dir_labels = os.path.join(output_dir_images, "labels")

# Ustvari izhodni mapi, če ne obstajata
os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(validation_dir_images, exist_ok=True)
os.makedirs(output_dir_labels, exist_ok=True)

# Ciljna velikost
target_size = (256, 256, 256)

# Prehodi skozi vse datoteke v mapi
for filename in os.listdir(input_dir_images):
    if filename.endswith(".nii.gz"):
        input_filepath = os.path.join(input_dir_images, filename)
        
        # Preveri, ali datoteka sodi v "validation" ali ne
        if "0011" in filename or "0012" in filename or "0013" in filename:
            output_filepath = os.path.join(validation_dir_images, filename)
            print(f"Kopiram originalno sliko v validation: {filename}")

            # Nalaganje in shranjevanje originalne slike
            image = sitk.ReadImage(input_filepath)
            sitk.WriteImage(image, output_filepath)

        else:
            # Shranjevanje slike s paddingom v mapo Data
            output_filepath = os.path.join(output_dirImages, filename)
            print(f"Procesiram in paddam sliko: {filename}")
            
            # Nalaganje slike
            image = sitk.ReadImage(input_filepath)
            
            # Dodajanje paddinga
            padded_image = add_padding_to_image(image, target_size=target_size)
            
            # Shranjevanje slike s paddingom
            sitk.WriteImage(padded_image, output_filepath)

# Procesiranje label
for filename in os.listdir(input_dir_labels):
    if filename.endswith(".nii.gz"):
        input_filepath = os.path.join(input_dir_labels, filename)
        output_filepath = os.path.join(output_dir_labels, filename)
        
        print(f"Procesiram in paddam label: {filename}")
        label = sitk.ReadImage(input_filepath)
        padded_label = add_padding_to_image(label, target_size=target_size)
        sitk.WriteImage(padded_label, output_filepath)


print("Procesiranje slik in label je končano.")