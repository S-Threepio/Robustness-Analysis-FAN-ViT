# Is Attentional Channel Processing Design Required? Comprehensive Analysis Of Robustness Between Vision Transformers And Fully Attentional Networks

## This repository is the official implementation of the following paper- https://arxiv.org/abs/2306.05495

## The Link to the paper: https://arxiv.org/abs/2306.05495

A robustness analysis of FAN ViT models using standard adversarial attacks

# Dataset Setup :
- Download the Imagenet-1K dataset from : https://image-net.org/download-images
- Arrange the images in a folder named "val" inside which each folder will have all the images of that particular class
- Refer this link for arranging images : https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
- This will be used as ground truth

# Running the Code :
1. conda create -n <env_name> python=3.9 anaconda
2. conda activate <env_name>
3. pip install -r requirements.txt
4. pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
5. python run_imagenet.py
