# Uporabimo uradno Python sliko
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
#FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
#FROM pytorch/pytorch:1.7.1-cpu

# Working directory
WORKDIR /workdir

# Install
RUN pip install numpy
RUN pip install nibabel
RUN pip install timm==0.5.4

COPY source source

WORKDIR /workdir/source

ENTRYPOINT ["python", "Code/Test_C2FViT_pairwise.py"]
CMD ["--modelpath", "Model/C2FViT_affine_COM_pairwise_stagelvl3_118000.pth",  "--fixed", "Data/image_B.nii.gz", "--moving", "Data/image_A.nii.gz"]