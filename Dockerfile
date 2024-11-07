# Uporabimo uradno Python sliko
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Working directory
WORKDIR /workdir

# Install
RUN pip install numpy
RUN pip install nibabel

COPY source source

WORKDIR /workdir/source

CMD ["python", "Code/Test_C2FViT_template_matching.py"]

