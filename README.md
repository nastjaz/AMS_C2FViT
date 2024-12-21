# AMS IZZIV - final report
Nastja Žgalin

Affine Medical Image Registration with Coarse-to-Fine Vision Transformer (C2FViT)

https://github.com/nastjaz/AMS_C2FViT.git


## Method Explanation
This is the official Pytorch implementation of "Affine Medical Image Registration with Coarse-to-Fine Vision Transformer" (CVPR 2022), written by Tony C. W. Mok and Albert C. S. Chung.

![plot](./source/Figure/overview.png?raw=true)

The Vision Transformer (ViT) learns the optimal affine transformation matrix to align fixed and moving grayscale images. The C2FViT model employs a multistage pipeline with convolutional patch embeddings and transformer encoder blocks to extract hierarchical features at progressively finer scales. Locality is enhanced through convolutional patch embeddings and depth-wise convolutions in feed-forward layers, while global connectivity is achieved via multi-head self-attention mechanisms.

To enhance flexibility and generalizability in affine image registration, C2FViT predicts geometric transformation parameters—translation, rotation, scaling, and shearing—instead of directly estimating the affine matrix. The affine matrix is derived as the product of these geometric transformation matrices, enabling seamless adaptation to other parametric registration methods, such as rigid registration, by omitting scaling and shearing components. Geometrical constraints reduce the search space, with rotation and shearing parameters restricted to [-π, +π], translation constrained to ±50% of spatial resolution, and scaling limited to [0.5, 1.5]. For rotation and shearing, the center of mass is used as the origin instead of the geometric center.

In the unsupervised learning paradigm, the model minimizes the dissimilarity between the fixed image 𝐹 and the warped moving image 𝑀(𝜑(𝐴𝑓)), where 𝐴𝑓 is the affine transformation matrix predicted by the model and 𝜑 represents the spatial transformation function. 
The similarity measure is based on the negative normalized cross-correlation (NCC), ensuring robust alignment between 𝐹 and 𝑀(𝜑(𝐴𝑓)). 


## Results

```bash
aggregated_results:
        LogJacDetStd        : 0.00000 +- 0.00000 | 30%: 0.00000
        TRE_kp              : 12.12685 +- 2.19995 | 30%: 13.76419
        TRE_lm              : 12.12293 +- 3.21796 | 30%: 13.88543
        DSC                 : 0.24743 +- 0.07594 | 30%: 0.18039
        HD95                : 47.75687 +- 10.91122 | 30%: 38.20628
```

![plot](./source/Figure/0011.png?raw=true)


## Docker Information

First command builds a Docker image named my-docker-image from a Dockerfile in the current directory (AMS_C2FViT).

`docker build -t my-docker-image -f Dockerfile .`


### Example
`docker run --name new-container --runtime=nvidia -it --rm -v $(pwd):/workdir --workdir /workdir/source my-docker-image python3 Code/Test_C2FViT_pairwise.py --modelpath Model/CBCT_affineC2FViT_1000stagelvl3_0.pth --fixed Data/validation/ThoraxCBCT_0011_0001.nii.gz --moving Data/validation/ThoraxCBCT_0011_0000.nii.gz`


## Data Preparation

The original images in our dataset have dimensions of 256x192x192. However, to train the model, the input images need to be resized to 256x256x256. This resizing is achieved through padding. Specifically, we use the `ImageResize.py` function, which adds padding to the images to ensure they reach the required dimensions of 256x256x256. The padded images are then saved in the `Data` directory, where they are ready for use in training the model.

The data preparation process begins with loading the original images (imagesTr) and labels (labelsTr) into the `source/OriginalData` and `source/OriginalLabels` directories. After processing the images and labels through the ImageResize.py function (which resizes the images and adds padding where necessary), the directory structure of the data is organized as follows: 

source/
├── Data/
│   ├── ThoraxCBCT_0000_*.nii.gz
│   ├── ThoraxCBCT_0001_*.nii.gz
│   ├── ...
│   ├── ThoraxCBCT_0010_*.nii.gz
│   ├── labels/
│   |    ├── ThoraxCBCT_0011_*.nii.gz
│   |    ├── ThoraxCBCT_0012_*.nii.gz
│   |    └── ThoraxCBCT_0013_*.nii.gz
│   └── validation/
│       ├── ThoraxCBCT_0011_*.nii.gz
│       ├── ThoraxCBCT_0012_*.nii.gz
│       └── ThoraxCBCT_0013_*.nii.gz


## Train Command

`docker run --name {name of a container} --runtime=nvidia -it --rm -v $(pwd):/workdir --workdir {workdir} --shm-size=8g my-docker-image python3 Train_C2FViT_pairwise.py --modelname {model_name} --lr 1e-4 --iteration 1000 --checkpoint 1000 --datapath {data_path} --com_initial True`

Once the training starts, files will be saved in two main directories within the `AMS_C2FViT` folder:
- `Log/`: This directory will contain logs from the training process. It helps to monitor the training progress and performance.
- `Model/`: This directory stores the trained model's weights and checkpoint files. 

After the training is complete, you need to copy the weights from the `Model/` directory to the `source/Model/` directory in your project structure for further use.

### Example

`docker run --name new-container --runtime=nvidia -it --rm -v $(pwd):/workdir --workdir /workdir/source --shm-size=8g my-docker-image python3 Code/Train_C2FViT_pairwise.py --modelname CBCT_affineC2FViT_10000 --lr 1e-4 --iteration 1000 --checkpoint 1000 --datapath Data --com_initial True`


## Test Command

`docker run --name {name of a container} --runtime=nvidia -it --rm -v $(pwd):/workdir --workdir {workdir} python3 Test_C2FViT_pairwise.py --modelpath {model_path} --fixed {fixed_img_path} --moving {moving_img_path}`

After running the test command, the output will be saved in two directories within the `AMS_C2FViT` folder:

- **Warped Image:** The warped (registered) image will be saved in the result folder.
- **Deformation Field:** The deformation field will be saved in the DeformationField folder.

### Example

`docker run --name new-container --runtime=nvidia -it --rm -v $(pwd):/workdir --workdir /workdir/source my-docker-image python3 Code/Test_C2FViT_pairwise.py --modelpath Model/CBCT_affineC2FViT_1000stagelvl3_0.pth --fixed OriginalData/validation/ThoraxCBCT_0011_0001.nii.gz --moving OriginalData/validation/ThoraxCBCT_0011_0000.nii.gz`


## Evaluation

`docker run --rm -u $UID:$UID -v /media/FastDataMama/nastjaz/AMS_C2FViT/DeformationField:/input -v ./output:/output/ gitlab.lst.fe.uni-lj.si:5050/domenp/deformable-registration python evaluation.py -v`