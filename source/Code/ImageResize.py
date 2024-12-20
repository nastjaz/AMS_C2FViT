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
input_dir = "/media/FastDataMama/nastjaz/AMS_C2FViT/source/OriginalData"
output_dir = "/media/FastDataMama/nastjaz/AMS_C2FViT/source/Data"

# Pot do map labels
# input_dir = "/media/FastDataMama/nastjaz/AMS_C2FViT/source/OriginalLabels"
# output_dir = "/media/FastDataMama/nastjaz/AMS_C2FViT/source/Data/labels"

# Ustvari izhodno mapo, če ne obstaja
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ciljna velikost
target_size = (256, 256, 256)

# Prehodi skozi vse datoteke v mapi
for filename in os.listdir(input_dir):
    if filename.endswith(".nii.gz"):
        input_filepath = os.path.join(input_dir, filename)
        output_filepath = os.path.join(output_dir, filename)

        print(f"Procesiram datoteko: {filename}")

        # Nalaganje slike
        image = sitk.ReadImage(input_filepath)

        # Dodajanje paddinga
        padded_image = add_padding_to_image(image, target_size=target_size)

        # Shranjevanje slike s paddingom
        sitk.WriteImage(padded_image, output_filepath)

        print(f"Shranjeno: {output_filepath}")

print("Procesiranje končano.")